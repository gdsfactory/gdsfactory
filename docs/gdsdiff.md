# gdsdiff

You can use the command line `pf diff gds1.gds gds2.gds` to see the difference between `gds1.gds` and `gds2.gds` files and show them in klayout.

For example, if you changed the mmi1x2 and made it 5um longer by mistake, you could `pf diff ref_layouts/mmi1x2.gds run_layouts/mmi1x2.gds` and see results of the GDS difference in Klayout

![git diff mmi](images/git_diff_gds_ex2.png)
