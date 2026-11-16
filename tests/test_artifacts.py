from dataclasses import asdict, replace
import inspect
import torch
import pytest
from tests.test_training import small_config
from nlm.model.nlm_model import NatureLearningModel
from nlm.training.dataset import SyntheticNatureDataset
from nlm.training.checkpoint import save_checkpoint, load_checkpoint, load_model, export_weights, load_pretrained

@pytest.mark.parametrize('suffix',['.pt','.safetensors'])
def test_nondefault_reconstruction_parity(tmp_path, small_config, suffix):
    config = replace(small_config, dropout=0.0, fusion_sparsity=0.5, ssm_conv_width=3,
                     quant_skip_modules=['unused_fixture_name'])
    model = NatureLearningModel(config).eval()
    path = tmp_path / ('weights'+suffix)
    export_weights(model,path,config)
    restored = load_model(path)
    assert asdict(restored.config) == asdict(config)
    assert not restored.training
    batch={k:v.unsqueeze(0) for k,v in SyntheticNatureDataset(1,config)[0].items() if k in inspect.signature(model.forward).parameters}
    with torch.no_grad():
        torch.testing.assert_close(model(**batch).next_state,restored(**batch).next_state,rtol=0,atol=0)

@pytest.mark.parametrize('damage',['missing_config','extra_config','bad_type','bad_version','preprocessor','missing_weight','extra_weight','shape','dtype','nan','legacy'])
def test_rejects_invalid_artifact_before_model_mutation(tmp_path,small_config,damage):
    model=NatureLearningModel(small_config)
    path=tmp_path/'checkpoint.pt'
    save_checkpoint(model,None,0,0.1,small_config,path)
    artifact=torch.load(path,weights_only=True)
    key=next(iter(artifact['model_state_dict']))
    if damage=='missing_config': artifact['config'].pop('dropout')
    elif damage=='extra_config': artifact['config']['unknown']=True
    elif damage=='bad_type': artifact['config']['hidden_dim']=True
    elif damage=='bad_version': artifact['artifact_version']=2
    elif damage=='preprocessor': artifact['preprocessor_version']='other'
    elif damage=='missing_weight': artifact['model_state_dict'].pop(key)
    elif damage=='extra_weight': artifact['model_state_dict']['unknown']=torch.ones(1)
    elif damage=='shape': artifact['model_state_dict'][key]=torch.zeros(1)
    elif damage=='dtype': artifact['model_state_dict'][key]=artifact['model_state_dict'][key].double()
    elif damage=='nan': artifact['model_state_dict'][key].fill_(float('nan'))
    elif damage=='legacy': artifact.pop('artifact_version')
    torch.save(artifact,path)
    before={k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError): load_checkpoint(path,model=model)
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)

def test_config_mismatch_and_permissive_load_rejected(tmp_path,small_config):
    model=NatureLearningModel(small_config); path=tmp_path/'model.pt'
    export_weights(model,path,small_config)
    other=NatureLearningModel(replace(small_config,dropout=0.2))
    with pytest.raises(ValueError,match='config differs'):load_checkpoint(path,other)
    with pytest.raises(ValueError,match='Permissive'):load_pretrained(model,path,strict=False)

def test_wrong_format_never_falls_back(tmp_path,small_config):
    model=NatureLearningModel(small_config)
    with pytest.raises(ValueError): export_weights(model,tmp_path/'weights.bin',small_config)
    path=tmp_path/'fake.safetensors'; torch.save({},path)
    with pytest.raises(Exception):load_model(path)

def test_safe_deserialization_rejects_pickle_object(tmp_path):
    path=tmp_path/'unsafe.pt';torch.save({'object':ValueError},path)
    with pytest.raises(Exception):load_model(path)
