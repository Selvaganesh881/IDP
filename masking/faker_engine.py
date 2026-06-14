from __future__ import annotations

import logging

from faker import Faker

from masking.recognizer import PIIEntity

logger = logging.getLogger(__name__)


class PIIFakerEngine:
    def __init__(self, locale: str = "en_US") -> None:
        # Initialize Faker with optional locale
        self.faker = Faker(locale)

        # Stateful inventory tracking to keep text masks deterministic across chunks/pages
        self.global_mapping: dict[str, str] = {}

        # Map Presidio/NER entity types directly to Faker generation methods
        self.provider_map = {
            "PERSON": self.faker.name,
            "EMAIL_ADDRESS": self.faker.company_email,
            "PHONE_NUMBER": self.faker.phone_number,
            "ORGANIZATION": self.faker.company,
            "LOCATION": self.faker.city,
            "IBAN_CODE": self.faker.iban,
            "CREDIT_CARD": self.faker.credit_card_number,
            "US_SSN": self.faker.ssn,
            "IP_ADDRESS": self.faker.ipv4,
            "URL": self.faker.url,
            # For dates, we use a lambda to format it consistently to string
            "DATE_TIME": lambda: self.faker.date_this_decade().strftime("%Y-%m-%d"),
        }

    def _generate_fake_value(self, entity_type: str, original_text: str) -> str:
        """
        Returns a consistent, highly realistic fake value for a given original string sequence.
        """
        cleaned_key = original_text.strip().lower()

        # 1. Return existing fake value if we've seen this exact entity before
        if cleaned_key in self.global_mapping:
            return self.global_mapping[cleaned_key]

        # 2. Generate a new realistic value based on the recognized type
        # If the type isn't in our map, fallback to a generic alphanumeric string
        generator = self.provider_map.get(
            entity_type, lambda: f"REF-{self.faker.lexify(text='????-????')}"
        )

        fake_value = str(generator())

        # 3. Store in dictionary to maintain global document consistency
        self.global_mapping[cleaned_key] = fake_value
        return fake_value

    def mask_text(self, text: str, entities: list[PIIEntity]) -> str:
        """
        Substitutes discovered sensitive words with realistic Faker entities.
        Processes items in reverse alignment order to prevent character displacement loops.
        """
        if not entities:
            return text

        # CRITICAL: Sort from tail to head to isolate downstream index shifting
        reverse_sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
        modified_text = text

        for entity in reverse_sorted_entities:
            fake_placeholder = self._generate_fake_value(
                entity.entity_type, entity.text
            )

            # Splice the realistic fake value cleanly into the markdown layer
            modified_text = (
                modified_text[: entity.start]
                + fake_placeholder
                + modified_text[entity.end :]
            )

        return modified_text
