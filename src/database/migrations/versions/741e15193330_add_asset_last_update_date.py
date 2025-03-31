"""[Add] Asset Last Update Date

Revision ID: 741e15193330
Revises: 663c6dd559f0
Create Date: 2025-03-31 00:45:38.461552

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '741e15193330'
down_revision: Union[str, None] = '663c6dd559f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('assets', sa.Column('last_update_date', sa.DateTime, nullable=True, default=None))


def downgrade() -> None:
    op.drop_column('assets', 'last_update_date')
