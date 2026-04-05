"""[Add] Valid Asset + change in PriceData

Revision ID: 140e02410495
Revises: 53036188f890
Create Date: 2025-04-09 00:26:10.177669

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '140e02410495'
down_revision: Union[str, None] = '53036188f890'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    
    # Assets Table
    op.add_column('assets', sa.Column('valid', sa.Boolean(), nullable=False, server_default='1'))
    
    # PriceData Table
    op.drop_constraint("pk_prices_data", "prices_data", type_="primary")
    # Very weird, but this is the only way to make this run
    op.execute("DROP VIEW IF EXISTS bb_ai7rtrsjk55a13mork6koie1ig_8fodgv_mig_cvqqt68uhbl60mimqn6g.prices_data CASCADE")
    op.drop_column('prices_data', 'open_status')
    op.drop_column('prices_data', 'price')
    op.add_column('prices_data', sa.Column('open_price', sa.Float(), nullable=False))
    op.add_column('prices_data', sa.Column('close_price', sa.Float(), nullable=False))
    op.create_primary_key(
        "pk_prices_data",
        "prices_data",
        ["date", "ticker"]
    )


def downgrade() -> None:
    
    # Assets Table
    op.drop_column('assets', 'valid')
    
    # PriceData Table
    op.drop_constraint("pk_prices_data", "prices_data", type_="primary")
    op.drop_column('prices_data', 'open_price')
    op.drop_column('prices_data', 'close_price')
    op.add_column('prices_data', sa.Column('price', sa.Float(), nullable=False))
    op.add_column('prices_data', sa.Column('open_status', sa.Boolean(), nullable=False))
    op.create_primary_key(
        "pk_prices_data",
        "prices_data",
        ["date", "ticker", "open_status"]
    )
