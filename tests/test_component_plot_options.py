"""Test Component.plot() pixel buffer options functionality."""

import gdsfactory as gf
from gdsfactory.typings import PixelBufferOptions


class TestComponentPlotOptions:
    """Test suite for Component.plot() pixel buffer options."""

    def test_plot_backward_compatibility(self):
        """Test that plot() still works without new parameters."""
        c = gf.components.straight(length=10, width=0.5)
        
        # Should work without any arguments
        c.plot()
        
        # Should work with return_fig=True
        fig = c.plot(return_fig=True)
        assert fig is not None

    def test_plot_with_empty_options(self):
        """Test plot() with empty pixel_buffer_options dict."""
        c = gf.components.straight(length=10, width=0.5)
        
        options: PixelBufferOptions = {}
        fig = c.plot(pixel_buffer_options=options, return_fig=True)
        assert fig is not None

    def test_plot_with_custom_dimensions(self):
        """Test plot() with custom width and height."""
        c = gf.components.straight(length=10, width=0.5)
        
        options: PixelBufferOptions = {
            "width": 1600,
            "height": 1200
        }
        fig = c.plot(pixel_buffer_options=options, return_fig=True)
        assert fig is not None
        
        # Figure size should be proportional to pixel dimensions
        size = fig.get_size_inches()
        assert size[0] > 0 and size[1] > 0

    def test_plot_with_all_options(self):
        """Test plot() with all available pixel buffer options."""
        c = gf.components.straight(length=10, width=0.5)
        
        options: PixelBufferOptions = {
            "width": 1024,
            "height": 768,
            "linewidth": 2,
            "oversampling": 1,
            "resolution": 1.5
        }
        fig = c.plot(pixel_buffer_options=options, return_fig=True)
        assert fig is not None

    def test_plot_with_partial_options(self):
        """Test plot() with only some options specified."""
        c = gf.components.straight(length=10, width=0.5)
        
        # Test with only width
        options_width: PixelBufferOptions = {"width": 1000}
        fig1 = c.plot(pixel_buffer_options=options_width, return_fig=True)
        assert fig1 is not None
        
        # Test with only oversampling
        options_oversample: PixelBufferOptions = {"oversampling": 2}
        fig2 = c.plot(pixel_buffer_options=options_oversample, return_fig=True)
        assert fig2 is not None

    def test_plot_different_resolutions_produce_different_sizes(self):
        """Test that different resolutions actually affect output."""
        c = gf.components.straight(length=10, width=0.5)
        
        # Default resolution
        fig_default = c.plot(return_fig=True)
        default_size = fig_default.get_size_inches()
        
        # High resolution
        options_high: PixelBufferOptions = {"width": 1600, "height": 1200}
        fig_high = c.plot(pixel_buffer_options=options_high, return_fig=True)
        high_size = fig_high.get_size_inches()
        
        # Low resolution  
        options_low: PixelBufferOptions = {"width": 400, "height": 300}
        fig_low = c.plot(pixel_buffer_options=options_low, return_fig=True)
        low_size = fig_low.get_size_inches()
        
        # Sizes should be different
        assert not (default_size == high_size).all()
        assert not (default_size == low_size).all()
        assert not (high_size == low_size).all()

    def test_plot_with_oversampling(self):
        """Test that oversampling parameter works."""
        c = gf.components.straight(length=10, width=0.5)
        
        for oversampling in [0, 1, 2, 3]:
            options: PixelBufferOptions = {
                "width": 800,
                "height": 600,
                "oversampling": oversampling
            }
            fig = c.plot(pixel_buffer_options=options, return_fig=True)
            assert fig is not None

    def test_plot_with_linewidth(self):
        """Test that linewidth parameter works."""
        c = gf.components.straight(length=10, width=0.5)
        
        for linewidth in [0, 1, 2, 3]:
            options: PixelBufferOptions = {
                "width": 800,
                "height": 600,
                "linewidth": linewidth
            }
            fig = c.plot(pixel_buffer_options=options, return_fig=True)
            assert fig is not None

    def test_plot_with_resolution_parameter(self):
        """Test that resolution parameter works."""
        c = gf.components.straight(length=10, width=0.5)
        
        for resolution in [0.0, 0.5, 1.0, 2.0]:
            options: PixelBufferOptions = {
                "width": 800,
                "height": 600,
                "resolution": resolution
            }
            fig = c.plot(pixel_buffer_options=options, return_fig=True)
            assert fig is not None

    def test_plot_none_options(self):
        """Test that None pixel_buffer_options works (should use defaults)."""
        c = gf.components.straight(length=10, width=0.5)
        
        fig = c.plot(pixel_buffer_options=None, return_fig=True)
        assert fig is not None