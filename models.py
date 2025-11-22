from __future__ import annotations

from typing import List

from pydantic import BaseModel


class NewsletterLink(BaseModel):
    title: str
    description: str
    url: str
    posted_by: str


class NewsletterPayload(BaseModel):
    links: List[NewsletterLink]
