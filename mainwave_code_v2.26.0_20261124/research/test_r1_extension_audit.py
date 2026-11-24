import pandas as pd

from research.r1_extension_audit import bucket, matched_near


def test_band_boundary_includes_exact_two_atr_in_the_baseline():
    assert [bucket(x) for x in (1.5,1.50001,2,2.00001,2.5,3,3.00001)]==[
        "below_1.5","1.5_to_2","1.5_to_2","2_to_2.5","2_to_2.5","2.5_to_3","above_3"]


def test_pairing_uses_same_day_board_liquidity_and_does_not_reuse_controls():
    values=[("near1","2_to_2.5",100,-4.,2.1,"STAR","2026-09-01"),
            ("near2","2_to_2.5",110,3.,2.2,"STAR","2026-09-01"),
            ("valid","1.5_to_2",105,2.,1.9,"STAR","2026-09-01"),
            ("wrong_liq","1.5_to_2",231,0.,1.8,"STAR","2026-09-01"),
            ("wrong_day","1.5_to_2",100,1.,1.8,"STAR","2026-09-02"),
            ("wrong_board","1.5_to_2",100,-1.,1.8,"MAIN","2026-09-01")]
    cols=["symbol","band","amount20","net","bias_atr","board","date"]
    f=pd.DataFrame(values,columns=cols).assign(mature=True,status="CLOSED",market="GREEN",
                                               risk_pct=3.,pulse_delay=1,opening_bias=2.,opening_gap_atr=0.)
    p=matched_near(f)
    assert list(p[["near","control","net_delta"]].itertuples(index=False,name=None))==[("near1","valid",-6.)]
    p=matched_near(f,fail_only=True)
    assert list(p[["near","control"]].itertuples(index=False,name=None))==[("near2","near1")]
