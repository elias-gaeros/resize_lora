from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass(slots=True)
class ModelContext:
    model_type: str
    components_present: Set[str]
    base_keys: Set[str]
    maps: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass(slots=True)
class MappingResult:
    canonical_key: str
    lora_key_base: str
    suffix: str
    matched_rule: str
