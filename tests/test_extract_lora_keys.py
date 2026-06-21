from extract_lora_keys import KeyFixture, detect_formats, select_diverse


def _fixture(family, source, fingerprint, formats):
    return KeyFixture(family, source, fingerprint, tuple(formats), (source,))


def test_detect_formats_covers_comfyui_and_peft_variants():
    assert detect_formats(
        [
            "lora_unet_x.lora_down.weight",
            "lora_unet_x.lora_up.weight",
            "lora_unet_y.lokr_w1",
            "diffusion_model.z.lora_A.weight",
            "diffusion_model.z.lora_proj_down",
        ]
    ) == ("lokr", "lora", "peft_lora", "svdquant_lora")


def test_selection_is_globally_duplicate_free_and_family_diverse():
    fixtures = [
        _fixture("large", "large/shared", "same", ["lora"]),
        _fixture("rare", "rare/shared", "same", ["lora"]),
        _fixture("large", "large/peft", "peft", ["peft_lora"]),
        _fixture("third", "third/loha", "loha", ["loha"]),
    ]

    selected = select_diverse(fixtures, limit=3)

    assert len({item.fingerprint for item in selected}) == len(selected)
    assert {item.family for item in selected} == {"large", "rare", "third"}
    assert next(item for item in selected if item.fingerprint == "same").family == "rare"
