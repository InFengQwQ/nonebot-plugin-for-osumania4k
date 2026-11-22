import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import nonebot

try:
    nonebot.get_driver()
except ValueError:
    nonebot.init()

from nonebot_plugin_rework.config import BUNDLED_PLUGIN_DIR, RenderConfig
from nonebot_plugin_rework.renderer import RenderError, Renderer


class RendererRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.renderer = Renderer(RenderConfig(plugin_dir=BUNDLED_PLUGIN_DIR, settle_ms=0))
        self.ready = {"overlayVisible": False, "status": "fixture", "star": "1"}
        self.renderer._wait_until_ready = AsyncMock(return_value=self.ready)
        self.browsers = []
        metrics = patch(
            "nonebot_plugin_rework.renderer._calculate_reference_metrics",
            return_value={"sunnyRework": None, "sunnyEstDiff": None, "osuSr": None},
        )
        metrics.start()
        self.addCleanup(metrics.stop)
        self.start_browser = AsyncMock(side_effect=self.install_browser)
        startup = patch.object(self.renderer, "_start_browser", self.start_browser)
        startup.start()
        self.addCleanup(startup.stop)

    async def asyncTearDown(self):
        await self.renderer.close()

    async def install_browser(self):
        card = SimpleNamespace(
            bounding_box=AsyncMock(return_value={"x": 0, "y": 0, "width": 100, "height": 100})
        )
        page = Mock()
        page.is_closed.return_value = False
        page.goto = AsyncMock()
        page.evaluate = AsyncMock(return_value=self.ready)
        page.wait_for_timeout = AsyncMock()
        page.query_selector = AsyncMock(return_value=card)
        page.screenshot = AsyncMock(return_value=b"PNG fixture")
        browser = Mock()
        browser.is_connected.return_value = True
        browser.close = AsyncMock()
        playwright = SimpleNamespace(stop=AsyncMock())
        self.renderer._page = page
        self.renderer._browser = browser
        self.renderer._playwright = playwright
        self.browsers.append((page, browser, playwright))

    async def test_driver_disconnect_rebuilds_before_retrying_goto(self):
        await self.install_browser()
        old_page, old_browser, old_driver = self.browsers[0]
        # The public health flags can still look healthy after the driver exits.
        old_page.goto.side_effect = Exception(
            "Page.goto: Connection closed while reading from the driver"
        )

        png = await self.renderer.render("current chart")

        self.assertEqual(png, b"PNG fixture")
        old_page.goto.assert_awaited_once()
        old_browser.close.assert_awaited_once()
        old_driver.stop.assert_awaited_once()
        self.start_browser.assert_awaited_once()
        self.browsers[1][0].goto.assert_awaited_once()
        self.assertEqual(self.renderer._state.osu_text, "current chart")

    async def test_closed_page_or_disconnected_browser_is_replaced(self):
        for resource in ("page", "browser"):
            with self.subTest(resource=resource):
                await self.install_browser()
                page, browser, driver = self.browsers[-1]
                if resource == "page":
                    page.is_closed.return_value = True
                else:
                    browser.is_connected.return_value = False

                await self.renderer.render("chart")

                page.goto.assert_not_awaited()
                browser.close.assert_awaited_once()
                driver.stop.assert_awaited_once()
                self.assertIsNot(self.renderer._page, page)
                await self.renderer._close_browser()

    async def test_healthy_browser_is_reused(self):
        await self.renderer.render("first chart")
        page = self.renderer._page

        await self.renderer.render("second chart")

        self.start_browser.assert_awaited_once()
        self.assertIs(self.renderer._page, page)
        self.assertEqual(page.goto.await_count, 2)

    async def test_repeated_failure_is_bounded_and_next_request_can_recover(self):
        async def broken_browser():
            await self.install_browser()
            self.renderer._page.goto.side_effect = Exception("driver disconnected")

        self.start_browser.side_effect = broken_browser
        with self.assertRaisesRegex(RenderError, "自动重建后仍未恢复"):
            await self.renderer.render("chart")

        self.assertEqual(self.start_browser.await_count, 2)
        self.assertIsNone(self.renderer._page)
        self.assertIsNone(self.renderer._browser)
        self.assertIsNone(self.renderer._playwright)
        for page, browser, driver in self.browsers:
            page.goto.assert_awaited_once()
            browser.close.assert_awaited_once()
            driver.stop.assert_awaited_once()

        self.start_browser.side_effect = self.install_browser
        self.assertEqual(await self.renderer.render("next chart"), b"PNG fixture")

    async def test_browser_close_error_does_not_skip_driver_or_http_cleanup(self):
        await self.install_browser()
        _, browser, driver = self.browsers[-1]
        browser.close.side_effect = Exception("connection already closed")
        server = Mock()
        self.renderer._server = server

        await self.renderer.close()

        driver.stop.assert_awaited_once()
        server.shutdown.assert_called_once()
        server.server_close.assert_called_once()
        self.assertIsNone(self.renderer._server)

    async def test_partial_startup_failure_releases_browser_and_driver(self):
        async def failed_startup():
            await self.install_browser()
            self.renderer._page = None
            raise RuntimeError("new_context failed")

        self.start_browser.side_effect = failed_startup
        with self.assertRaisesRegex(RuntimeError, "new_context failed"):
            await self.renderer._ensure_browser()

        _, browser, driver = self.browsers[-1]
        browser.close.assert_awaited_once()
        driver.stop.assert_awaited_once()
        self.assertIsNone(self.renderer._browser)
        self.assertIsNone(self.renderer._playwright)

    async def test_analysis_errors_do_not_restart_the_browser(self):
        self.renderer._wait_until_ready.side_effect = RenderError("invalid chart")

        with self.assertRaisesRegex(RenderError, "invalid chart"):
            await self.renderer.render("chart")

        self.start_browser.assert_awaited_once()
        self.browsers[0][1].close.assert_not_awaited()

    async def test_cancelled_startup_releases_partial_resources(self):
        async def cancelled_startup():
            await self.install_browser()
            raise asyncio.CancelledError()

        self.start_browser.side_effect = cancelled_startup
        with self.assertRaises(asyncio.CancelledError):
            await self.renderer._ensure_browser()

        _, browser, driver = self.browsers[-1]
        browser.close.assert_awaited_once()
        driver.stop.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
