import model
import json
import jax
import jax.numpy as jnp
import flax
import tokenizer


def download_files():
    from huggingface_hub import hf_hub_download
    import safetensors.numpy
    
    config_file = hf_hub_download("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "open_clip_config.json")
    params_file = hf_hub_download("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "open_clip_model.safetensors")
    with open(config_file, "r") as f:
        config = json.load(f)
    params = model.reformat_params(safetensors.numpy.load_file(params_file))
    return config, params

def verify_parameter_shapes(config: dict, params: dict):

    def init():
        model_def = model.CLIPCfg(**config['model_cfg']).build()
        example_input = jnp.ones((1, 224, 224, 3))
        example_text = jnp.ones((1, 77), dtype=jnp.int32)
        return model_def.init(jax.random.PRNGKey(0), example_input, example_text, train=False)['params']

    expected_params = jax.eval_shape(init)

    def compare(old, new):
        old = flax.traverse_util.flatten_dict(old)
        new = flax.traverse_util.flatten_dict(new)

        old_keys = set(old.keys())
        new_keys = set(new.keys())
        assert old_keys == new_keys, f'{old_keys-new_keys=} {new_keys-old_keys=}'

        mismatched_shapes = {k: (new[k].shape, old[k].shape) for k in old_keys & new_keys if new[k].shape != old[k].shape}
        assert not mismatched_shapes, f'{mismatched_shapes=}'


    compare(expected_params, params)


def check_logits(config: dict, params: dict):
    from PIL import Image
    import requests
    import numpy as np

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    raw_image = np.asarray(Image.open(requests.get(url, stream=True).raw).resize((224, 224)).convert('RGB'))

    simple_tokenizer = tokenizer.SimpleTokenizer()
    text = simple_tokenizer(["a diagram", "a dog", "a cat"]).numpy()
    model_def = model.CLIPCfg(**config['model_cfg']).build()
    logits = model_def.apply({'params': params}, raw_image[None], text, train=False)
    assert jnp.allclose(logits, np.array([[19.3143, 14.1324, 23.5409]]), atol=0.1), logits

if __name__ == '__main__':
    config, params = download_files()
    verify_parameter_shapes(config, params)
    check_logits(config, params)
