# `KeyMapper` Design and Current Status

## Purpose

`loralib.key_mapper.KeyMapper` maps adapter tensor names to tensor names that actually
exist in a supplied base-model safetensors file. It builds aliases once per base model
and then performs dictionary lookups for individual adapter keys.

This component is experimental. It is used by `test_key_mapper.py`; it is not yet the
mapper used by `resize_lora.py`.

## Build Process

`KeyMapper(base_model_path)` performs these steps:

1. Read the base model's tensor names with `safetensors.safe_open`.
2. Infer a model type and component set from key-prefix heuristics in
   `KeyMapper._detect_model_type_and_components()`.
3. Run the ordered generators in `DEFAULT_GENERATORS`.
4. Merge each generator's aliases into `final_mapping`.
5. Reject an alias if two generators map it to different canonical keys.

The current generators cover:

- ComfyUI/kohya-style UNet prefixes and direct base-model stems.
- SD/SDXL/SD3.5 CLIP-L and CLIP-G aliases.
- Selected FLUX and Chroma DiT block names.
- Selected SD3.5 joint-block and T5 names.
- Diffusers UNet down, mid, and up block names.
- LyCORIS-prefixed aliases derived from earlier generators.

Generator order matters because `LyCORISPrefixGenerator` consumes the mapping built by
the preceding generators.

## Lookup Process

`map_from_lora(raw_key)`:

1. Removes the longest matching suffix from `KeyMapper.SUFFIXES`.
2. Looks up the remaining module name in `final_mapping`.
3. Returns `MappingResult` on success or `None` otherwise.

Longest-first suffix matching is a correctness requirement. For example,
`.hada_w1_a.weight` must be tested before generic `.weight`, or a LoHa key is split at
the wrong boundary.

## Conflict Handling

`KeyMapper._merge_mappings()` rejects conflicting aliases produced by different
generators. A generator still returns a normal dictionary, so conflicting assignments
made internally by one generator cannot be recovered after the dictionary has already
overwritten a value. New generators should detect ambiguity while constructing their
own result.

## Known Limitations

- Model identification is inline heuristic code, not a separate `ModelIdentifier`
  service or signature registry.
- The canonical key is a real key from the supplied checkpoint. It is not a universal
  architecture-independent representation.
- Mapping support does not imply resize support. The main resize path remains focused
  on SDXL LoRA/LoCon.
- Several architecture conversions are partial and must be evaluated against an
  appropriate base checkpoint.
- GLoRA/Klein `.R` and normalization `.scale` tensors are not currently recognized.
- The mapping harness measures name coverage, not numerical adapter equivalence.

## Extension Checklist

When adding a naming convention:

1. Add a focused `MappingGenerator` or extend the narrowest applicable generator.
2. Only emit canonical names present in `ModelContext.base_keys`.
3. Detect ambiguous aliases rather than relying on dictionary overwrite order.
4. Add synthetic tests for every supported suffix and alias form.
5. Run the corpus workflow in `data_driven_dev.md` against a matching base checkpoint.
