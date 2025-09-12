"""Test Component.plot() pixel buffer options functionality."""

import pytest

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import PixelBufferOptions


class TestComponentPlot:
    """Test suite for Component.plot()."""

    @pytest.fixture(scope="class")
    def component_under_test(self) -> Component:
        return gf.components.straight(length=10, width=0.5)

    def test_plot_no_options(self, component_under_test: Component) -> None:
        # Should work without any arguments
        component_under_test.plot()

        # Should work with return_fig=True
        fig = component_under_test.plot(return_fig=True)
        assert fig is not None

    def test_plot_with_empty_pixel_buffer_options(
        self, component_under_test: Component
    ) -> None:
        """Test plot() with empty pixel_buffer_options dict."""
        options: PixelBufferOptions = {}
        fig = component_under_test.plot(pixel_buffer_options=options, return_fig=True)
        assert fig is not None

    def test_plot_with_custom_dimensions(self, component_under_test: Component) -> None:
        """Test plot() with custom width and height."""
        options: PixelBufferOptions = {"width": 1600, "height": 1200}
        fig = component_under_test.plot(pixel_buffer_options=options, return_fig=True)
        assert fig is not None

        # Figure size should be proportional to pixel dimensions
        size = fig.get_size_inches()
        assert size[0] > 0 and size[1] > 0

    def test_plot_with_all_pixel_buffer_options(
        self, component_under_test: Component
    ) -> None:
        """Test plot() with all available pixel buffer options."""
        options: PixelBufferOptions = {
            "width": 1024,
            "height": 768,
            "linewidth": 2,
            "oversampling": 1,
            "resolution": 1.5,
        }
        fig = component_under_test.plot(pixel_buffer_options=options, return_fig=True)
        assert fig is not None

    def test_plot_different_resolutions_produce_different_sizes(
        self, component_under_test: Component
    ) -> None:
        """Test that different resolutions actually affect output."""
        # Default resolution
        fig_default = component_under_test.plot(return_fig=True)
        default_size = fig_default.get_size_inches()

        # High resolution
        options_high: PixelBufferOptions = {"width": 1600, "height": 1200}
        fig_high = component_under_test.plot(
            pixel_buffer_options=options_high, return_fig=True
        )
        high_size = fig_high.get_size_inches()

        # Low resolution
        options_low: PixelBufferOptions = {"width": 400, "height": 300}
        fig_low = component_under_test.plot(
            pixel_buffer_options=options_low, return_fig=True
        )
        low_size = fig_low.get_size_inches()

        # Sizes should be different
        assert not (default_size == high_size).all()
        assert not (default_size == low_size).all()
        assert not (high_size == low_size).all()
