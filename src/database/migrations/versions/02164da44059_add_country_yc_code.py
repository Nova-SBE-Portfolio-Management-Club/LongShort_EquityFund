"""[Add] country yc_code

Revision ID: 02164da44059
Revises: 12d47eafc111
Create Date: 2025-04-30 12:19:21.068994

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '02164da44059'
down_revision: Union[str, None] = '12d47eafc111'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('countries', sa.Column('yc_code', sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column('countries', 'yc_code')
