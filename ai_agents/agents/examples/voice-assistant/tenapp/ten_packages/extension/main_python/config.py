from pydantic import BaseModel, Field, field_validator


class MainControlConfig(BaseModel):
    greeting: str = "Hello, I am your AI assistant."
    flush_destinations: list[str] = Field(
        default_factory=lambda: ["agora_rtc"], min_length=1
    )

    @field_validator("flush_destinations")
    @classmethod
    def normalize_flush_destinations(cls, destinations: list[str]) -> list[str]:
        normalized = [destination.strip() for destination in destinations]
        if any(not destination for destination in normalized):
            raise ValueError("flush destinations must be non-empty names")
        return list(dict.fromkeys(normalized))
