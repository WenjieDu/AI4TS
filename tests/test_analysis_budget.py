"""Source-bound explicit slow-host analysis budget; original 60s remains intact."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pandas as pd
import pytest

from ai4ts.data_protocols.analysis import create_analysis_plan, validate_analysis_plan


ROLES = dict(sample_id_col="id", timestamp_col="time", feature_cols=["x"], label_col=None)


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "source.csv"
    pd.DataFrame(dict(id=["A"]*48, time=pd.date_range("2024-01-01",periods=48,freq="1h"),
                      x=np.linspace(1.,3.,48))).to_csv(p,index=False)
    return p


def test_legacy_default_still_sealed_and_explicit_slow_host_bound(source):
    baseline=create_analysis_plan(source,roles=ROLES,methods=["acf"])
    assert baseline["status"]=="ready",baseline
    assert baseline["budget"]["compute_seconds"]==60
    assert "diagnostic_compute_seconds" not in baseline["request"]
    assert validate_analysis_plan(baseline,source)==baseline
    other=create_analysis_plan(source,roles=ROLES,methods=["acf"],diagnostic_compute_seconds=120)
    assert other["status"]=="ready",other
    assert other["budget"]["compute_seconds"]==120
    assert other["request"]["diagnostic_compute_seconds"]==120
    assert other["sha256"]!=baseline["sha256"]
    assert validate_analysis_plan(other,source)==other
    same60=create_analysis_plan(source,roles=ROLES,methods=["acf"],diagnostic_compute_seconds=60)
    assert same60["status"]=="ready"
    assert same60["budget"]["compute_seconds"]==60
    assert same60["sha256"]!=baseline["sha256"]
    assert source.is_file()


@pytest.mark.parametrize("invalid",[True,False,0,1,59,61,90,119,121,180,-120,120.0,float("inf"),"120",[120],{"seconds":120}])
def test_invalid_budget_is_not_admitted(source,invalid):
    result=create_analysis_plan(source,roles=ROLES,methods=["acf"],diagnostic_compute_seconds=invalid)
    assert result["status"]=="blocked",result
    assert "budget" not in result


def test_diagnostic_budget_rejected_on_forecast_even_when_int(source):
    plan=create_analysis_plan(source,"forecast-baselines",ROLES,targets=["x"],
       split_policy="chronological",diagnostic_compute_seconds=120)
    assert plan["status"]=="blocked",plan


@pytest.mark.parametrize("field,value",[("budget",121),("request",121),("request",60),("budget",60)])
def test_tampering_with_budget_or_request_is_rejected(source,field,value):
    original=create_analysis_plan(source,roles=ROLES,methods=["acf"],diagnostic_compute_seconds=120)
    assert original["status"]=="ready"
    tampered=copy.deepcopy(original)
    if field=="request":tampered["request"]["diagnostic_compute_seconds"]=value
    else:tampered["budget"]["compute_seconds"]=value
    with pytest.raises(ValueError,match="changed|ready|parameters"):
        validate_analysis_plan(tampered,source)
    assert validate_analysis_plan(original,source)==original
