"""[Add] Foregin Key Country-Company

Revision ID: 663c6dd559f0
Revises: 63c9dc00c90c
Create Date: 2025-03-31 00:32:37.063605

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '663c6dd559f0'
down_revision: Union[str, None] = '63c9dc00c90c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_foreign_key(
        constraint_name="companies_cntry_fkey",
        source_table="companies",
        referent_table="countries",
        local_cols=["country"],
        remote_cols=["name"],
        ondelete="SET NULL"  
    )


def downgrade() -> None:
    op.drop_constraint("companies_cntry_fkey", "companies", type_="foreignkey")
