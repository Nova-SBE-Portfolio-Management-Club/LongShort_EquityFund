"""[Add] Futures Macro Miscell Classes

Revision ID: 114a61cd9569
Revises: 140e02410495
Create Date: 2025-04-11 23:25:23.234843

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '114a61cd9569'
down_revision: Union[str, None] = '140e02410495'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Futures table
    op.create_table(
        'futures',
        sa.Column('ticker', sa.String(15), sa.ForeignKey('assets.ticker', ondelete="CASCADE"), primary_key=True),
        sa.Column('bloomberg_ticker', sa.String(20)),
    )

def downgrade() -> None:
    # Drop tables
    op.drop_table('futures')

