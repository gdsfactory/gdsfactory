import gdsfactory as gf
from gdsfactory.generic_tech import LAYER
from gdsfactory.typings import LayerSpec

from gpdk import cross_section
from gpdk.components import (
    bend_euler,
    mmi1x2,
    mmi2x2,
    straight,
)
from gpdk.components.pads import bump_pad_grid

to_um = gf.kcl.to_um


@gf.cell
def flat_det() -> gf.Component:
    c = gf.Component()
    det = c << gf.components.ge_detector_straight_si_contacts()
    c.add_ports(ports=det.ports)
    c.flatten()
    return c


@gf.cell
def det_two_ports() -> gf.Component:
    c = gf.Component()
    det1 = c << flat_det()
    det2 = c << flat_det()
    det1.name = "DET_1"
    det2.name = "DET_2"
    det1.movey(-15)
    det2.movey(15)
    c.add_port(port=det1.ports["o1"])
    c.add_port(port=det2.ports["o1"])
    c.auto_rename_ports()
    c.add_port(port=det1.ports["bot"], name="bot_1")
    c.add_port(port=det2.ports["bot"], name="bot_2")
    c.add_port(port=det1.ports["top"], name="top_1")
    c.add_port(port=det2.ports["top"], name="top_2")

    return c


@gf.cell
def transceiver_mzi(fill: bool = True) -> gf.Component:
    c = gf.Component()

    xs = cross_section.strip(radius=50)
    bend = bend_euler(cross_section=xs)
    br = abs(bend.ports["o1"].dx - bend.ports["o2"].dx)

    def route_bundle(
        ports1: list[gf.kf.Port | gf.Port],
        ports2: list[gf.kf.Port | gf.Port],
        steps: list[dict[str, float]] | None = None,
        bboxes: list[gf.kdb.Box] | None = None,
        end_straight_length: float = 0,
        start_straight_length: float = 0,
        separation: float = 9,
        sort_ports: bool = False,
    ) -> list[gf.kf.routing.generic.ManhattanRoute]:
        return gf.routing.route_bundle(
            c,
            ports1,
            ports2,
            straight=straight,
            bboxes=bboxes,
            separation=separation,
            end_straight_length=end_straight_length,
            start_straight_length=start_straight_length,
            sort_ports=sort_ports,
            steps=steps,
            cross_section=xs,
        )

    def route_elec(
        ports1: list[gf.kf.Port | gf.Port],
        ports2: list[gf.kf.Port | gf.Port],
        steps: list[dict[str, float]] | None = None,
        bboxes: list[gf.kdb.Box] | None = None,
        end_straight_length: float = 0,
        start_straight_length: float = 0,
        separation: float = 9,
        sort_ports: bool = False,
        start_angles: int | list[int] | None = None,
        end_angles: int | list[int] | None = None,
        layer: LayerSpec = LAYER.M3,
    ) -> list[gf.kf.routing.generic.ManhattanRoute]:
        if layer == LAYER.M1:
            return gf.routing.route_bundle_electrical(
                c,
                ports1,
                ports2,
                bboxes=bboxes,
                separation=separation,
                end_straight_length=end_straight_length,
                start_straight_length=start_straight_length,
                sort_ports=sort_ports,
                steps=steps,
                end_angles=end_angles,
                start_angles=start_angles,
                auto_taper=False,
                allow_width_mismatch=True,
                radius=13,
                cross_section=cross_section.metal1(layer=LAYER.M1),
            )
        elif layer == LAYER.M2:
            return gf.routing.route_bundle_electrical(
                c,
                ports1,
                ports2,
                bboxes=bboxes,
                separation=separation,
                end_straight_length=end_straight_length,
                start_straight_length=start_straight_length,
                sort_ports=sort_ports,
                steps=steps,
                end_angles=end_angles,
                start_angles=start_angles,
                auto_taper=False,
                allow_width_mismatch=True,
                radius=20,
                cross_section=cross_section.metal2(layer=LAYER.M2),
            )
        else:
            return gf.routing.route_bundle_electrical(
                c,
                ports1,
                ports2,
                bboxes=bboxes,
                separation=separation,
                end_straight_length=end_straight_length,
                start_straight_length=start_straight_length,
                sort_ports=sort_ports,
                steps=steps,
                end_angles=end_angles,
                start_angles=start_angles,
                auto_taper=False,
                allow_width_mismatch=True,
                radius=30,
                cross_section=cross_section.metal3(layer=LAYER.M3),
            )

    eams: list[gf.ComponentReference] = []
    srcs: list[gf.ComponentReference] = []
    mzm_pins: list[gf.ComponentReference] = []
    src_det_pins: list[gf.ComponentReference] = []
    mzm_det_pins: list[gf.ComponentReference] = []
    br_det_pins: list[gf.ComponentReference] = []
    ps_te_pins: list[gf.ComponentReference] = []
    ps_tm_pins: list[gf.ComponentReference] = []
    d_pd_pins: list[gf.ComponentReference] = []
    m_pd_pins: list[gf.ComponentReference] = []
    ecc_mpd_pins: list[gf.ComponentReference] = []
    lb_mpd_pins: list[gf.ComponentReference] = []
    mpd_fine_pins: list[gf.ComponentReference] = []

    amzi_pos = [(-400, -100), (350, 100), (-400, 400), (350, 600)]

    for i in range(4):
        mzm = c << gf.components.mzm(cross_section=xs, length_x=0.5e3)
        mzm.name = f"MZM_{i}"
        mzm.transform(gf.kdb.DTrans(rot=2, mirrx=True, x=4500, y=-1600 + i * 930))
        eams.append(mzm)
        mzm_pins.append([p for p in mzm.ports if p.port_type == "electrical"])
        src = c << gf.components.grating_coupler_elliptical()
        src.name = f"SRC_{i}"
        src.move((2400, -2000 + i * 920))
        srcs.append(src)
        arc = c << bend_euler(cross_section=xs, angle=180)
        arc.connect("o2", src, "o1")
        src_tap = c << gf.components.mmi2x2()
        src_tap.name = f"SRC_TAP_{i}"
        src_tap.connect("o1", arc, "o1")
        src_det = c << flat_det()
        src_det.name = f"SRC_DET_{i}"
        trl_bb = src.dbbox()
        src_det.transform(
            gf.kdb.DTrans(rot=2, x=trl_bb.right - 300, y=trl_bb.bottom - 10)
        )
        src_det_pins.append([src_det.ports["bot"], src_det.ports["top"]])

        route_bundle(
            [src_tap.ports["o4"]], [src_det.ports["o1"]], start_straight_length=20
        )
        mzm_tap = c << gf.components.mmi2x2(cross_section=xs)
        mzm_tap.name = f"MZM_TAP_{i}"
        mzm_tap.move(
            (mzm_tap.ports["o1"].dx, mzm_tap.ports["o4"].dy),
            (src_tap.ports["o1"].dx, mzm.ports["o1"].dy),
        )
        route_bundle(
            ports1=[mzm.ports["o2"], mzm.ports["o1"]],
            ports2=[mzm_tap.ports["o4"], src_tap.ports["o3"]],
            bboxes=[mzm.bbox()],
        )
        mzm_det = c << flat_det()
        mzm_det.name = f"MZM_DET_{i}"
        mzm_det.transform(gf.kdb.DTrans(rot=2, x=mzm_tap.dx - 370, y=mzm_tap.dy - 210))
        mzm_det_pins.append([mzm_det.ports["bot"], mzm_det.ports["top"]])
        route_bundle(
            [mzm_tap.ports["o1"]],
            [mzm_det.ports["o1"]],
        )

        mzi_1 = c << gf.components.mzi1x2_2x2(
            delta_length=60, length_y=40, cross_section=xs
        )
        mzi_1.name = f"MZI1_{i}"
        x, y = amzi_pos[i]
        mzi_1.transform(gf.kdb.DTrans(rot=2, mirrx=True, x=-4000 + x, y=-2000 + y))

        route_bundle(
            [mzm_tap.ports["o2"]],
            [mzi_1.ports["o1"]],
            steps=[
                {"x": 785 - i * 18, "dy": -150},
                {"y": -2420 + i * 18},
                {"x": -3550 + i * 18},
                {"dy": 10},
            ],
            end_straight_length=i * 9,
        )

        tap1 = c << gf.components.mmi2x2_with_sbend()
        tap1.name = f"TAP1_{i}"
        tap1.move((-4500, -1100 - i * 34))

        route_bundle(
            [mzi_1.ports["o2"]],
            [tap1.ports["o2"]],
            steps=[{"x": -5400 + i * 36, "dy": 68.25}, {"dy": 60}],
        )
        br_det = c << flat_det()
        br_det.name = f"BR_DET_{i}"
        br_det.move((1100, -1400 + i * 930))
        br_det_pins.append([br_det.ports["bot"], br_det.ports["top"]])

        amzi_bb = mzi_1.dbbox()
        if not i % 2:
            route_bundle(
                [tap1.ports["o1"]],
                [br_det.ports["o1"]],
                steps=[
                    {"x": -5386 + i * 36, "dy": -br},
                    {"y": amzi_bb.top + 20},
                    {"x": amzi_bb.right + 10},
                    {"y": amzi_bb.center().y + 30},
                    # {"x": amzi_bb.right + 20},
                    {"x": -3541 + i * 18},
                    {"y": -2411 + i * 18},
                    {"x": 776 - i * 18},
                    {"dy": 10},
                ],
            )
        else:
            route_bundle(
                [tap1.ports["o1"]],
                [br_det.ports["o1"]],
                steps=[
                    {"x": -5386 + i * 36, "dy": -br},
                    {"y": amzi_bb.center().y + 30},
                    {"x": amzi_bb.left - 10},
                    {"y": amzi_bb.top + 10},
                    # {"x": amzi_bb.right + 20},
                    {"x": -3541 + i * 18},
                    {"y": -2411 + i * 18},
                    {"x": 776 - i * 18},
                    {"dy": 10},
                ],
            )

        tx_gca = c << gf.components.grating_coupler_array(n=2)
        tx_gca.name = f"TX_GCA_{i}"
        tx_gca.transform(gf.kdb.DTrans(1, True, x=-4500, y=-750 + i * 200))
        rx_gca = c << gf.components.grating_coupler_array(n=2)
        rx_gca.name = f"RX_GCA_{i}"
        rx_gca.transform(gf.kdb.DTrans(1, True, x=-4500, y=650 - i * 200))

        route_bundle(
            [tap1.ports["o4"]],
            [tx_gca.ports["o1"]],
            steps=[{"dx": br + 9 + i * 27, "dy": br}, {"dy": 20}],
        )

        dbr_te = c << gf.components.dbr_tapered()
        dbr_te.name = f"DBR_TE_{i}"
        dbr_tm = c << gf.components.dbr_tapered()
        dbr_tm.name = f"DBR_TM_{i}"
        dbr_tm.dmirror_y(0)  # Mirror about x-axis (equivalent to DTrans.M0)
        soa_te_bbox = dbr_te.dbbox()
        soa_tm_bbox = dbr_tm.dbbox()

        dbr_te.move((150 - soa_te_bbox.left, 1225 - i * 1050 - soa_te_bbox.bottom))
        dbr_tm.move((150 - soa_tm_bbox.left, 925 - i * 1050 - soa_tm_bbox.bottom))
        ps_te = c << gf.components.mzi2x2_2x2_phase_shifter(cross_section=xs)
        ps_te.name = f"PS_TE_{i}"
        ps_te.move((-2430, dbr_te.ports["o1"].dy - ps_te.ports["o4"].dy))
        ps_tm = c << gf.components.mzi2x2_2x2_phase_shifter(cross_section=xs)
        ps_tm.name = f"PS_TM_{i}"
        ps_tm.move((-2430, dbr_tm.ports["o1"].dy - ps_tm.ports["o3"].dy))
        ps_te_pins.append([p for p in ps_te.ports if p.port_type == "electrical"])
        ps_tm_pins.append([p for p in ps_tm.ports if p.port_type == "electrical"])

        route_bundle(
            [ps_te.ports["o4"], ps_tm.ports["o3"]],
            [dbr_te.ports["o1"], dbr_tm.ports["o1"]],
            separation=7,
        )

        dpd1 = c << flat_det()
        dpd2 = c << flat_det()
        dpd1.name = f"DET_TE_{i}"
        dpd2.name = f"DET_TM_{i}"
        dpd1.transform(
            gf.kdb.DTrans(rot=1, mirrx=True, x=dbr_te.dx - 585 + i * 600, y=2500)
        )
        dpd2.transform(
            gf.kdb.DTrans(rot=1, mirrx=True, x=dbr_tm.dx - 555 + i * 600, y=2500)
        )
        dpd1.dbbox() + dpd2.dbbox()
        dpd_te_route, dpd_tm_route = route_bundle(
            [dbr_te.ports["o2"], dbr_tm.ports["o2"]],
            [dpd1.ports["o1"], dpd2.ports["o1"]],
            separation=7,
            steps=[
                {"dx": br + 33.75 + i * 45, "dy": br + 3.75},
                {"y": 1691},
            ],
            end_straight_length=54 if i in [0, 3] else 0,
        )

        mpd_det = c << det_two_ports()
        mpd_det.name = f"RXSW_MPD_{i}"
        x = dbr_te.dx
        if i < 3:
            mpd_det.transform(gf.kdb.DTrans(rot=1, x=x - 640 - (3 - i) * 175, y=2500))
        else:
            mpd_det.transform(
                gf.kdb.DTrans(rot=1, mirrx=True, x=dbr_te.dx + 1500, y=2500)
            )
        d_pd_pins.append([dpd1.ports["bot"], dpd1.ports["top"]])
        d_pd_pins.append([dpd2.ports["bot"], dpd2.ports["top"]])
        m_pd_pins.append(
            [
                mpd_det.ports["bot_1"],
                mpd_det.ports["top_1"],
                mpd_det.ports["bot_2"],
                mpd_det.ports["top_2"],
            ]
        )

        steps_te: list[dict[str, float]] = [
            {"x": x + br + 54 + i * 45, "dy": 150},
        ]
        match i:
            case 3:
                steps_te.extend(
                    [
                        {"y": 2395.5},
                        {"x": x + 800},
                        {"y": 2552},
                        {"x": mpd_det.dx + 100},
                    ]
                )
            case 2:
                steps_te.extend(
                    [
                        {"y": 2597},
                        {"x": x - 656},
                        {"y": mpd_det.dbbox().bottom - 9},
                    ]
                )
            case 1:
                steps_te.extend(
                    [
                        {"y": 2413.5},
                        {"x": x - 200},
                        {"y": 2570},
                        {"x": x - 629},
                        {"y": mpd_det.dbbox().bottom - 36},
                    ]
                )
            case 0:
                steps_te = []
        steps_tm: list[dict[str, float]] = [
            {"x": dbr_tm.dx + br + 81.5 + i * 45, "dy": br},
        ]
        match i:
            case 3:
                steps_tm.extend([{"y": 2359.5}, {"dx": br}])
            case 2:
                steps_tm.extend(
                    [
                        {"y": 2413.5},
                        {"x": x + 782},
                        {"y": 2606},
                        {"x": x - 665},
                        {"y": mpd_det.dbbox().bottom},
                    ]
                )
            case 1:
                steps_tm.extend(
                    [
                        {"y": 2579},
                        {"x": x - 638},
                        {"y": mpd_det.dbbox().bottom - 27},
                    ]
                )
            case 0:
                steps_tm.extend(
                    [
                        {"y": 2395.5},
                        {"x": x - 218},
                        {"y": 2552},
                        {"x": x - 611},
                        {"y": mpd_det.dbbox().bottom - 54},
                    ]
                )

        route_bundle(
            [ps_te.ports["o3"]],
            [mpd_det.ports["o2"]],
            steps=steps_te,
        )
        route_bundle(
            [ps_tm.ports["o4"]],
            [mpd_det.ports["o1"]],
            steps=steps_tm,
        )

        route_diff = (
            abs(ps_tm.ports["o1"].dy - rx_gca.ports["o0"].dy)
            - abs(ps_te.ports["o2"].dy - rx_gca.ports["o1"].dy)
            + to_um(dpd_tm_route.length_straights)
            - to_um(dpd_te_route.length_straights)
            + 20
        )
        rd2 = route_diff / 2
        steps_tm = [
            {"dx": -br - 10, "dy": br},
            {"dy": br},
            {"dx": 2 * br - min(rd2, 0)},
            {"dy": 2 * br},
            {"dx": -2 * br - 10 + min(rd2, 0)},
            {"dy": -4 * br},
            {"dx": -10},
        ]
        steps_te = [
            {"dx": -br - 30, "dy": -br},
            {"dy": -3 * br},
            {"dx": -2 * br - max(rd2, 0) - 10},
            {"dy": 2 * br},
            {"dx": 2 * br + max(rd2, 0)},
            {"dy": 2 * br},
            {"dx": -10},
        ]
        # steps_tm = []
        dtm = 9 if i == 0 else 1000
        dte = 0 if i == 0 else 1009
        route_tm = route_bundle(
            [ps_tm.ports["o2"]],
            [rx_gca.ports["o0"]],
            start_straight_length=0,
            steps=steps_tm,
            end_straight_length=dtm + (3 - i) * 54,
        )
        route_te = route_bundle(
            [ps_te.ports["o1"]],
            [rx_gca.ports["o1"]],
            start_straight_length=0,
            steps=steps_te,
            end_straight_length=dte + (3 - i) * 54,
        )
        print(
            f"{route_diff=}\n",
            f"{route_te[0].length_straights=}\n",
            f"{dpd_te_route.length_straights=}\n",
            f"{route_tm[0].length_straights=}\n",
            f"{dpd_tm_route.length_straights=}\n",
            f"route_te = {route_te[0].length_straights + dpd_te_route.length_straights}\n",
            f"route_tm = {route_tm[0].length_straights + dpd_tm_route.length_straights}\n",
            f"{(route_te[0].length_straights + dpd_te_route.length_straights) - (route_tm[0].length_straights + dpd_tm_route.length_straights)=}",
        )

        fine_tap = c << mmi2x2()
        fine_tap.name = f"LB_FINE_TAP_{i}"
        lb_spl = c << mmi1x2()
        lb_spl.name = f"LB_SPL_{i}"
        sw_bb = ps_te.dbbox(LAYER.WG)
        ft_bb = fine_tap.dbbox(LAYER.WG)
        coarse_amzi = c << gf.components.mzi2x2_2x2(cross_section=xs)
        coarse_amzi.name = f"COARSE_AMZI_{i}"
        coarse_amzi.transform(
            gf.kdb.DTrans(x=-460, y=260) * ps_te.dtrans * gf.kdb.DTrans.M0
        )
        if i != 0:
            fine_tap.transform(
                gf.kdb.DTrans(sw_bb.p2 - ft_bb.p1 + gf.kdb.DVector(800, 285 + 8.125))
            )
            lb_spl.dcplx_trans = (
                coarse_amzi.ports["o1"].dcplx_trans
                * gf.kdb.DCplxTrans(mirrx=True, rot=180, x=70)
                * lb_spl.ports["o3"].dcplx_trans.inverted()
            )
            coarse_bb = coarse_amzi.dbbox()
            route_bundle(
                [lb_spl.ports["o2"]],
                [fine_tap.ports["o2"]],
                steps=[{"dx": 2 * br, "y": coarse_bb.top + 30}, {"x": coarse_bb.right}],
            )
            mpd_fine = c << flat_det()
            mpd_fine.name = f"LB_FINE_MPD_{i}"
            mpd_fine.transform(gf.kdb.DTrans.R180)
            mpd_fine.transform(fine_tap.dtrans)
            mpd_fine.transform(gf.kdb.DTrans(u=gf.kdb.DVector(100, -150)))
            route_bundle(
                [fine_tap.ports["o4"]],
                [mpd_fine.ports["o1"]],
            )
            route_bundle(
                [coarse_amzi.ports["o1"]],
                [lb_spl.ports["o3"]],
            )
            mpd_fine_pins.append([mpd_fine.ports["bot"], mpd_fine.ports["top"]])

        else:
            fine_tap.transform(gf.kdb.DTrans(rot=1, x=tx_gca.ports["o1"].dx, y=1111))
            lb_spl.transform(
                gf.kdb.DTrans(u=(rx_gca.dbbox().p2 + gf.kdb.DVector(-500, 100)).to_v())
            )
            mpd_fine = c << flat_det()
            mpd_fine.name = f"LB_FINE_MPD_{i}"
            mpd_fine.transform(
                gf.kdb.DTrans(
                    rot=1, x=-2300, y=mpd_det.dbbox().top - mpd_fine.dbbox().right
                )
            )
            mpd_fine_pins.append([mpd_fine.ports["bot"], mpd_fine.ports["top"]])
            route_bundle(
                [fine_tap.ports["o4"]],
                [mpd_fine.ports["o1"]],
                start_straight_length=700,
            )
            route_bundle(
                [lb_spl.ports["o2"]],
                [fine_tap.ports["o1"]],
            )
            route_bundle(
                [mzi_1.ports["o3"]],
                [lb_spl.ports["o1"]],
                steps=[
                    {"x": c.dbbox().left - 9, "dy": br},
                    {"y": tap1.dbbox().top + 9},
                ],
            )
            route_bundle(
                [lb_spl.ports["o3"]],
                [coarse_amzi.ports["o1"]],
                start_straight_length=500,
            )

        lb_sw = c << gf.components.mzi1x2_2x2(cross_section=xs)
        lb_sw.name = f"LB_SW_{i}"
        lb_sw.transform(
            gf.kdb.DTrans(x=370, y=-130) * coarse_amzi.dtrans * gf.kdb.DTrans.M90
        )
        route_bundle(
            [coarse_amzi.ports["o3"]],
            [lb_sw.ports["o1"]],
        )
        lbin_spl = c << mmi1x2()
        lbin_spl.name = f"LBIN_SPL_{i}"
        lbin_spl.connect("o1", lb_sw.ports["o3"])

        # comment out for error in connectivity
        lbin_spl.movex(-100)
        route_bundle([lbin_spl.ports["o1"]], [lb_sw.ports["o3"]])
        # comment out

        ecc_mpd = c << flat_det()
        ecc_mpd.name = f"ECC_MPD_{i}"
        ecc_mpd.transform(
            # gf.kdb.DCpTrans(u=(fine_tap.dbbox().p1 + gf.kdb.DVector(x=0, y=-885)))
            dbr_tm.ports["o1"].dcplx_trans
            * gf.kdb.DCplxTrans(x=-100, y=-121.002)
            * ecc_mpd.ports["o1"].dcplx_trans.inverted()
        )
        ecc_mpd_pins.append([ecc_mpd.ports["bot"], ecc_mpd.ports["top"]])
        steps = [
            {"dx": br + 0.001 + i * 27, "dy": br + 0.001},
            {"y": tx_gca.dbbox().bottom - 9},
            {"x": tx_gca.dbbox().left - (4 - i) * 18},
            {"y": rx_gca.dbbox().bottom - 9},
            (
                {"x": to_um(route_tm[0].backbone[-2].x) - 18}
                if i > 0
                else {"x": to_um(route_tm[0].backbone[-2].x) + 18}
            ),
            {"y": ecc_mpd.ports["o1"].dy},
            {"dx": 10},
        ]
        route_bundle(
            [tap1.ports["o3"]],
            [ecc_mpd.ports["o1"]],
            steps=steps,
            end_straight_length=1600,
        )
        rx_bb = rx_gca.dbbox()
        lb_tm = route_bundle(
            [lbin_spl.ports["o3"]],
            [ps_tm.ports["o1"]],
            steps=[
                {"dx": -2 * br, "dy": 2 * br},
                {
                    "x": to_um(route_te[0].backbone[-2].x) + (9 if i > 0 else -9),
                },
                {"y": rx_bb.top + 9},
                {"x": rx_bb.left - 18},
                {"y": rx_bb.bottom - 5},
                {"x": to_um(route_tm[0].backbone[-2].x) - (9 if i > 0 else -9)},
                {"y": ps_tm.ports["o2"].dy - 2 * br - 10},
                {"dx": 2 * br},
                {"dy": 2 * br},
                {"dx": 2 * br},
                {"dy": -2 * br},
                {"dx": br},
            ],
        )[0]

        base_te_length = (
            3255.977 if i != 0 else 4455.977
        )  # base length of the TM that is relevant,
        # this can be calculated by having the base path length `lb_tm.length_straights` to be subtracted from the lb_te length
        route_diff = c.kcl.to_um(lb_tm.length_straights) - base_te_length - 40

        lb_te = route_bundle(
            [lbin_spl.ports["o2"]],
            [ps_te.ports["o2"]],
            steps=[
                {
                    "dx": -br - 0.001,
                    "dy": -br,
                },
                {"y": ps_te.ports["o1"].dy + 18},
                {"x": ps_te.ports["o1"].dx + 10},
                {"dy": 2 * br + 10},
                {"dx": 699 + 2 * br + route_diff / 2},
                {"dy": -2 * br - 9},
                {"dx": -2 * br},
                {"dy": 2 * br},
                {"dx": -690 - route_diff / 2},
                {"dy": -2 * br - 10},
                {"dx": -800 if i != 0 else -1400},
                {"dy": 2 * br},
                {"dx": -br},
            ],
        )[0]
        print(f"{lb_te.n_bend90=} {lb_te.length_straights=}")
        print(f"{lb_tm.n_bend90=} {lb_tm.length_straights=}")
        print(f"{lb_tm.length_straights - lb_te.length_straights=}")

        if i > 0:
            lb_coarse_mpd = c << flat_det()
            lb_coarse_mpd.name = f"LB_COARSE_MPD_{i}"
            lb_coarse_mpd.connect("o1", coarse_amzi, "o4", allow_width_mismatch=True)
            lb_coarse_mpd.movex(1250)
            route_bundle(
                [coarse_amzi.ports["o4"]],
                [lb_coarse_mpd.ports["o1"]],
            )
            lb_sw_mpd = c << flat_det()
            lb_sw_mpd.name = f"LB_SW_MPD_{i}"
            lb_sw_mpd.dtrans = gf.kdb.DTrans(y=-31) * lb_coarse_mpd.dtrans
            lb_mpd_pins.append([lb_coarse_mpd.ports["bot"], lb_coarse_mpd.ports["top"]])
            lb_mpd_pins.append([lb_sw_mpd.ports["bot"], lb_sw_mpd.ports["top"]])
            route_bundle(
                [lb_sw.ports["o2"]],
                [lb_sw_mpd.ports["o1"]],
                steps=[
                    {"dy": -2 * br - 10, "dx": -br},
                    {"dx": 755 / 2},
                    {"dy": 2 * br + 9},
                    {"dx": 2 * br},
                ],
            )
            route_bundle(
                [mzi_1.ports["o3"]],
                [lb_spl.ports["o1"]],
                steps=[
                    {"x": -5409 + i * 36, "dy": br},
                    {"y": tap1.ports["o1"].dy + 9},
                    {"x": -4182.393 - 256.606 + i * 27},
                    {"y": tx_gca.dbbox().bottom - 18},
                    {"x": -4809 + 241.805 - (3 - i) * 18},
                    {"y": rx_gca.dbbox().top + 18},
                    {"x": to_um(route_te[0].backbone[-2].x) + (18 if i > 0 else 0)},
                    {"y": lb_spl.ports["o1"].dy},
                    {"dx": br},
                ],
            )
        else:
            lb_coarse_mpd = c << flat_det()
            lb_coarse_mpd.name = f"LB_COARSE_MPD_{i}"
            lb_coarse_mpd.dtrans = mpd_fine.dtrans * gf.kdb.DTrans(y=-150, x=0)
            route_bundle([coarse_amzi.ports["o4"]], [lb_coarse_mpd.ports["o1"]])

            lb_sw_mpd = c << flat_det()
            lb_sw_mpd.name = f"LB_SW_MPD_{i}"
            lb_sw_mpd.dtrans = mpd_fine.dtrans * gf.kdb.DTrans(y=-300, x=0)

            lb_mpd_pins.append([lb_coarse_mpd.ports["bot"], lb_coarse_mpd.ports["top"]])
            lb_mpd_pins.append([lb_sw_mpd.ports["bot"], lb_sw_mpd.ports["top"]])

            route_bundle(
                [lb_sw.ports["o2"]],
                [lb_sw_mpd.ports["o1"]],
                steps=[
                    {"dx": -br - 0.001, "dy": -br},  # FIXME: rounding
                    {"dy": -br - 10},
                    {"x": ps_te.ports["o1"].dx},
                    {"dy": 2 * br + 20},
                    {"dx": 50},
                ],
            )

    power_comb_gc1 = c << gf.components.grating_coupler_array(n=2)
    power_comb_gc1.name = "TP_GC_0123_WG0"
    power_comb_gc1.rotate(90, center=(0, 0))
    power_comb_gc1.move((-5000, 1500))

    power_comb_gc2 = c << gf.components.grating_coupler_array(n=2)
    power_comb_gc2.name = "TP_GC_0123_WG1"
    power_comb_gc2.rotate(90, center=(0, 0))
    power_comb_gc2.move((-5000, 1300))

    power_comb1 = c << mmi2x2()
    power_comb1.name = "PWR_COMB_01"
    power_comb2 = c << mmi2x2()
    power_comb2.name = "PWR_COMB_23"
    power_comb1.move((-4700, 1750))
    power_comb2.move((-4700, 1770))

    route_bundle(
        [
            power_comb1.ports["o1"],
            power_comb1.ports["o2"],
            power_comb2.ports["o1"],
            power_comb2.ports["o2"],
        ],
        [
            power_comb_gc1.ports["o0"],
            power_comb_gc1.ports["o1"],
            power_comb_gc2.ports["o0"],
            power_comb_gc2.ports["o1"],
        ],
        separation=0.75,
        bboxes=[power_comb_gc1.bbox() + power_comb_gc2.bbox()],
        sort_ports=True,
    )
    route_bundle(
        [c.insts["LB_FINE_TAP_0"].ports["o3"]],
        [power_comb1.ports["o4"]],
    )
    route_bundle(
        [c.insts["LB_FINE_TAP_1"].ports["o3"]],
        [power_comb1.ports["o3"]],
        steps=[
            {"x": dbr_te.dx + br + 90.5, "dy": br},
            {"y": 2404.5},
            {"x": -34},
            {"y": 2561.5},
            {"x": -445},
            {"y": 2385},
            {"x": -850},
            {"y": 2550},
            {"dx": -10},
        ],
        end_straight_length=18,
    )
    route_bundle(
        [c.insts["LB_FINE_TAP_2"].ports["o3"]],
        [power_comb2.ports["o4"]],
        steps=[
            {"x": dbr_te.dx + br + 135.5, "dy": br},
            {"y": 2588.5},
            {"x": -472},
            {"y": 2412},
            {"x": -675},
            {"y": 2559},
            {"dx": -10},
        ],
        end_straight_length=9,
    )
    route_bundle(
        [c.insts["LB_FINE_TAP_3"].ports["o3"]],
        [power_comb2.ports["o3"]],
        steps=[
            {"x": dbr_te.dx + br + 180.5, "dy": br},
            {"y": 2404.5},
            {"x": 966},
            {"y": 2615},
            {"dx": -10},
        ],
    )

    if fill:
        gf.kf.utils.fill_tiled(
            c=c,
            fill_cell=gf.components.octagon(layer=LAYER.M3, side_length=30),
            fill_regions=[(gf.kdb.Region(c.bbox()), 0)],
            exclude_layers=[
                (get_layer_info(LAYER.M1), 100),
                (get_layer_info(LAYER.M2), 100),
                (get_layer_info(LAYER.WG), 10),
            ],
            row_step=gf.kdb.DVector(250, 0),
            col_step=gf.kdb.DVector(0, 250),
            tile_size=(1000, 1000),
            tile_border=(50, 50),
            n_threads=1,
        )

        # add bump fills
        bump_fill = c << bump_pad_grid(
            columns=41,
            rows=32,
            column_pitch=150,
            row_pitch=150,
            offset=0,
            add_via=True,
            skip_pads=[
                (12, 4),
                (12, 6),
                (12, 11),
                (12, 13),
                (12, 18),
                (12, 20),
                (12, 25),
                (12, 27),
                (19, 7),
                (19, 14),
                (19, 21),
                (20, 7),
                (20, 14),
                (20, 21),
                (35, 9),
                (35, 15),
                (35, 21),
            ],
        )
        bump_fill.x -= 2025 + 2100
        bump_fill.y -= 2287 + 450
        flat_te = [x for sub in ps_te_pins for x in sub]
        ps_te_pins_for_routing = [
            flat_te[i] for i in range(0, len(flat_te)) if i % 8 == 1 or i % 8 == 7
        ]
        flat_tm = [x for sub in ps_tm_pins for x in sub]
        ps_tm_pins_for_routing = [
            flat_tm[i] for i in range(0, len(flat_tm)) if i % 8 == 1 or i % 8 == 7
        ]
        flat_src_pd = [x for sub in src_det_pins for x in sub]
        flat_src_pd.sort(key=lambda port: port.y)
        flat_dpd = [x for sub in d_pd_pins for x in sub]
        flat_dpd.sort(key=lambda port: port.y)
        flat_mpd = [x for sub in m_pd_pins for x in sub]
        flat_mpd.sort(key=lambda port: port.y)
        flat_lbpd = [x for sub in lb_mpd_pins for x in sub]
        flat_lbpd.sort(key=lambda port: port.y)
        flat_fine = [x for sub in mpd_fine_pins for x in sub]
        flat_fine.sort(key=lambda port: port.y)
        ecc_mpd_flat = [x for sub in ecc_mpd_pins for x in sub]
        ecc_mpd_flat.sort(key=lambda port: port.y)
        flat_mzm_pd = [x for sub in mzm_det_pins for x in sub]
        flat_mzm_pd.sort(key=lambda port: port.y)
        br_det_flat = [x for sub in br_det_pins for x in sub]
        br_det_flat.sort(key=lambda port: port.y)
        total_pd_pins = (
            br_det_flat
            + flat_mzm_pd
            + flat_src_pd
            + flat_dpd
            + flat_mpd
            + flat_lbpd
            + flat_fine
            + ecc_mpd_flat
        )
        total_pd_pins.sort(key=lambda port: port.x)
        flat_mzm_pins = [x for sub in mzm_pins for x in sub if x.orientation == 180.0]
        flat_mzm_pins.sort(key=lambda port: port.y)

        # route the PD pins to bumps
        route_elec(
            ports1=[total_pd_pins[j] for j in range(0, 3)],
            ports2=[bump_fill.ports[f"e{28 - j}_11_e1"] for j in range(0, 3)],
            separation=8,
            start_straight_length=0,
            end_straight_length=0,
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(3, 6)],
            ports2=[bump_fill.ports[f"e{28 - j}_11_e1"] for j in range(3, 6)],
            separation=8,
            start_straight_length=90,
            end_straight_length=0,
            steps=[{"dx": -710}, {"dy": -1600}, {"dx": 120}],
            layer=LAYER.M3,
        )
        for i in range(3):
            route_elec(
                ports1=[total_pd_pins[j + 4 * i] for j in range(6, 8)],
                ports2=[
                    bump_fill.ports[f"e{7 + 7 * i - j}_20_e3"] for j in range(0, 2)
                ],
                separation=5,
                start_straight_length=0,
                end_straight_length=0,
                layer=LAYER.M3,
            )
            route_elec(
                ports1=[total_pd_pins[j + 4 * i] for j in range(8, 10)],
                ports2=[
                    bump_fill.ports[f"e{10 + 7 * i - j}_20_e3"] for j in range(0, 2)
                ],
                separation=5,
                start_straight_length=0,
                end_straight_length=0,
                layer=LAYER.M3,
            )
            route_elec(
                ports1=[total_pd_pins[j + 2 * i] for j in range(18, 19)],
                ports2=[
                    bump_fill.ports[f"e{7 + 7 * i + 2 * j}_21_e1"] for j in range(0, 1)
                ],
                separation=0,
                start_straight_length=10,
                end_straight_length=10,
                layer=LAYER.M3,
            )
            route_elec(
                ports1=[total_pd_pins[j + 2 * i] for j in range(19, 20)],
                ports2=[
                    bump_fill.ports[f"e{7 + 7 * i + 2 * j}_21_e1"] for j in range(1, 2)
                ],
                separation=0,
                start_straight_length=10,
                end_straight_length=10,
                layer=LAYER.M3,
            )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(24, 27)],
            ports2=[bump_fill.ports[f"e{28 - j}_19_e1"] for j in range(0, 3)],
            separation=8,
            start_straight_length=0,
            end_straight_length=0,
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(27, 30)],
            ports2=[bump_fill.ports[f"e{28 - j}_19_e1"] for j in range(3, 6)],
            separation=5,
            start_straight_length=90,
            end_straight_length=0,
            steps=[{"dx": -680}, {"dy": -1600}, {"dx": 120}],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(30, 33)],
            ports2=[bump_fill.ports[f"e{28 - j}_19_e1"] for j in range(6, 9)],
            separation=8,
            start_straight_length=135,
            end_straight_length=0,
            steps=[{"dx": -990}, {"dy": -2050}, {"dx": 120}],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(33, 36)],
            ports2=[bump_fill.ports[f"e{28 - j}_19_e1"] for j in range(9, 12)],
            separation=5,
            start_straight_length=190,
            end_straight_length=0,
            steps=[{"dx": -1300}, {"dy": -2500}, {"dx": 120}],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(36, 38)],
            ports2=[bump_fill.ports[f"e{28 - j}_21_e1"] for j in range(0, 2)],
            separation=5,
            start_straight_length=0,
            end_straight_length=0,
            steps=[{"dx": -40}, {"dy": -700}, {"dx": -120}],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(38, 40)],
            ports2=[bump_fill.ports[f"e{28 - j}_21_e1"] for j in range(2, 4)],
            separation=5,
            start_straight_length=30,
            end_straight_length=0,
            steps=[{"dx": -160}, {"dy": -550}, {"dx": -800}, {"dy": -900}, {"dx": 120}],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(40, 42)],
            ports2=[bump_fill.ports[f"e{28 - j}_22_e1"] for j in range(0, 2)],
            separation=5,
            start_straight_length=0,
            end_straight_length=0,
            steps=[{"dx": -40}, {"dy": -850}, {"dx": -120}],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(42, 44)],
            ports2=[bump_fill.ports[f"e{25 + j}_22_e3"] for j in range(0, 2)],
            separation=5,
            start_straight_length=0,
            end_straight_length=0,
            steps=[{"dx": 80}, {"dy": -1000}, {"dx": -1000}],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(52, 54)],
            ports2=[bump_fill.ports[f"e{23 + j}_22_e3"] for j in range(0, 2)],
            separation=5,
            start_straight_length=0,
            end_straight_length=0,
            steps=[
                {"dx": -330},
                {"dy": -1150},
                {"dx": -1200},
                {"dy": -600},
                {"dx": -120},
            ],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(54, 56)],
            ports2=[bump_fill.ports[f"e{21 + j}_22_e3"] for j in range(0, 2)],
            separation=5,
            start_straight_length=0,
            end_straight_length=0,
            steps=[
                {"dx": 60},
                {"dy": -500},
                {"dx": -280},
                {"dy": -800},
                {"dx": -1200},
                {"dy": -750},
                {"dx": -120},
            ],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(44, 52)],
            ports2=[
                bump_fill.ports[f"e{4 + int(j / 2) * 7}_{32 - (j % 2)}_e4"]
                for j in range(0, 8)
            ],
            separation=5,
            start_straight_length=0,
            end_straight_length=0,
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(56, 64)],
            ports2=[
                bump_fill.ports[f"e{8 - int(j / 4) - (j % 2) + int(j / 2) * 7}_36_e3"]
                for j in range(0, 8)
            ],
            separation=5,
            start_straight_length=0,
            end_straight_length=0,
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(64, 66)],
            ports2=[bump_fill.ports[f"e{17 + j}_22_e3"] for j in range(0, 2)],
            separation=5,
            start_straight_length=0,
            end_straight_length=0,
            steps=[
                {"dx": -330},
                {"dy": -1600},
                {"dx": -1350},
                {"dy": -750},
                {"dx": -300},
                {"dy": -300},
                {"dx": -120},
            ],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(66, 68)],
            ports2=[bump_fill.ports[f"e{19 + j}_22_e3"] for j in range(0, 2)],
            separation=5,
            start_straight_length=50,
            end_straight_length=0,
            steps=[
                {"dx": -500},
                {"dy": -1450},
                {"dx": -1350},
                {"dy": -750},
                {"dx": -300},
                {"dy": -300},
                {"dx": -120},
            ],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(68, 70)],
            ports2=[bump_fill.ports[f"e{13 + j}_22_e3"] for j in range(0, 2)],
            separation=5,
            start_straight_length=0,
            end_straight_length=0,
            steps=[
                {"dx": -150},
                {"dy": -1900},
                {"dx": -750},
                {"dy": -750},
                {"dx": -600},
                {"dy": -600},
                {"dx": -120},
            ],
            layer=LAYER.M3,
        )
        route_elec(
            ports1=[total_pd_pins[j] for j in range(70, 72)],
            ports2=[bump_fill.ports[f"e{15 + j}_22_e3"] for j in range(0, 2)],
            separation=5,
            start_straight_length=50,
            end_straight_length=0,
            steps=[
                {"dx": -210},
                {"dy": -500},
                {"dx": -130},
                {"dy": -1250},
                {"dx": -750},
                {"dy": -750},
                {"dx": -600},
                {"dy": -600},
                {"dx": -120},
            ],
            layer=LAYER.M3,
        )
        for i in range(0, 4):
            route_elec(
                ports1=[total_pd_pins[j + 2 * i] for j in range(72, 74)],
                ports2=[
                    bump_fill.ports[f"e{6 + j + 5 * i}_38_e3"] for j in range(0, 2)
                ],
                separation=5,
                start_straight_length=0,
                end_straight_length=10,
                steps=[{"dy": -100 - 180 * i}, {"dx": -100}],
                layer=LAYER.M3,
            )
            route_elec(
                ports1=[total_pd_pins[j + 2 * i] for j in range(80, 82)],
                ports2=[
                    bump_fill.ports[f"e{4 + j + 5 * i}_38_e3"] for j in range(0, 2)
                ],
                separation=5,
                start_straight_length=0,
                end_straight_length=10,
                steps=[{"dy": -170 - 180 * i if i < 3 else -540}, {"dx": -100}],
                layer=LAYER.M3,
            )
        # route the mzm to bumps
        for i in range(0, 8):
            route_elec(
                ports1=[flat_mzm_pins[j + 2 * i] for j in range(0, 2)],
                ports2=[
                    bump_fill.ports[f"e4_{13 + j + 2 * i}_e4"] for j in range(0, 2)
                ],
                separation=10,
                start_straight_length=0,
                end_straight_length=0,
                steps=[
                    {"dx": -100 - 40 * i},
                    {"y": -3100 + 40 * i},
                    {"x": -2100 + 300 * i},
                    {"dy": 300},
                ],
                layer=LAYER.M3,
            )
        # route the mzi heaters to bumps
        for j in range(0, 4):
            route_elec(
                ports1=[ps_te_pins_for_routing[i + 2 * j] for i in range(0, 1)],
                ports2=[
                    bump_fill.ports[f"e{28 - i - 7 * j}_12_e2"] for i in range(0, 1)
                ],
                separation=10,
                start_straight_length=0,
                end_straight_length=0,
                layer=LAYER.M3,
            )
            route_elec(
                ports1=[ps_te_pins_for_routing[i + 2 * j] for i in range(1, 2)],
                ports2=[
                    bump_fill.ports[f"e{28 - i - 7 * j}_12_e2"] for i in range(1, 2)
                ],
                separation=10,
                start_straight_length=55,
                end_straight_length=0,
                layer=LAYER.M3,
            )
            route_elec(
                ports1=[ps_tm_pins_for_routing[i + 2 * j] for i in range(0, 1)],
                ports2=[
                    bump_fill.ports[f"e{26 - i - 7 * j}_12_e2"] for i in range(0, 1)
                ],
                separation=10,
                start_straight_length=0,
                end_straight_length=0,
                layer=LAYER.M3,
            )
            route_elec(
                ports1=[ps_tm_pins_for_routing[i + 2 * j] for i in range(1, 2)],
                ports2=[
                    bump_fill.ports[f"e{26 - i - 7 * j}_12_e2"] for i in range(1, 2)
                ],
                separation=10,
                start_straight_length=55,
                end_straight_length=0,
                layer=LAYER.M3,
            )

    return c


def get_layer_info(layer: gf.LayerEnum) -> gf.kdb.LayerInfo:
    return gf.kcl.layout.get_info(layer)
