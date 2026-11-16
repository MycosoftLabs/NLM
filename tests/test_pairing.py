from dataclasses import replace
from datetime import datetime,timedelta,timezone
import pytest
import torch
from nlm.training.pairing import PairingSpec,ObservedFrame,ObservedPairDataset,build_observed_pairs
T=datetime(2026,1,1,tzinfo=timezone.utc)
SPEC=PairingSpec(('temperature','humidity'),('normalized','normalized'),'fixture-train-only-v1',60,0,
                 T+timedelta(seconds=180),T+timedelta(seconds=360),T+timedelta(seconds=600))
def frame(n,series='site-a'):
    return ObservedFrame(f'{series}-{n}',series,T+timedelta(seconds=n),{'temperature':0.,'humidity':n/1000},
                         {'temperature':'normalized','humidity':'normalized'},SPEC.normalizer_id)

def test_observed_zero_labels_and_split_purging():
    result=build_observed_pairs([frame(n) for n in range(0,601,60)],SPEC)
    assert ('site-a-120','split_boundary') in result.rejected
    assert ('site-a-300','split_boundary') in result.rejected
    assert ('site-a-540','split_boundary') in result.rejected
    assert result.pairs[0].target == (0.,.06)
    used={name:set() for name in ['train','validation','test']}
    for pair in result.pairs: used[pair.split].update([pair.input_id,pair.target_id])
    assert not used['train'] & used['validation']; assert not used['validation'] & used['test']

@pytest.mark.parametrize('damage',['missing','nan','overflow','unit','normalizer','naive','duplicate_id','duplicate_time'])
def test_preflight_rejects_bad_cohort(damage):
    rows=[frame(0),frame(60)]
    if damage=='missing':rows[1]=replace(rows[1],values={'temperature':0.})
    if damage=='nan':rows[1]=replace(rows[1],values={'temperature':0.,'humidity':float('nan')})
    if damage=='overflow':rows[1]=replace(rows[1],values={'temperature':0.,'humidity':1e100})
    if damage=='unit':rows[1]=replace(rows[1],units={'temperature':'C','humidity':'normalized'})
    if damage=='normalizer':rows[1]=replace(rows[1],normalizer_id='fitted-on-test')
    if damage=='naive':rows[1]=replace(rows[1],measured_at=datetime(2026,1,1))
    if damage=='duplicate_id':rows[1]=replace(rows[1],frame_id=rows[0].frame_id)
    if damage=='duplicate_time':rows[1]=replace(rows[1],measured_at=rows[0].measured_at)
    with pytest.raises(ValueError):build_observed_pairs(rows,SPEC)

def test_no_cross_series_or_current_state_fallback():
    result=build_observed_pairs([frame(0),frame(60,'site-b')],SPEC)
    assert not result.pairs;assert len(result.rejected)==2

def test_ambiguous_tolerance_rejected_and_nearest_supported():
    spec=replace(SPEC,tolerance_seconds=10)
    assert ('site-a-0','ambiguous_target') in build_observed_pairs([frame(0),frame(50),frame(70)],spec).rejected
    assert build_observed_pairs([frame(0),frame(51),frame(70)],spec).pairs[0].target_id=='site-a-51'

def test_adapter_preserves_input_and_attaches_future_observation():
    result=build_observed_pairs([frame(0),frame(60)],SPEC)
    inputs=[{'spatial_features':torch.ones(37)},{'spatial_features':torch.zeros(37)}]
    ds=ObservedPairDataset(inputs,['site-a-0','site-a-60'],result,'train',2)
    assert len(ds)==1;assert ds[0]['spatial_features'].sum()==37
    torch.testing.assert_close(ds[0]['next_state_target'],torch.tensor([0.,.06]))
    assert 'next_state_target' not in inputs[0]
    with pytest.raises(ValueError):ObservedPairDataset(inputs,['site-a-0','site-a-60'],result,'train',14)

def test_invalid_spec():
    for spec in [replace(SPEC,tolerance_seconds=60),replace(SPEC,channels=('temperature','temperature')),
                 replace(SPEC,normalizer_id=''),replace(SPEC,validation_end=SPEC.train_end)]:
        with pytest.raises(ValueError):build_observed_pairs([],spec)
