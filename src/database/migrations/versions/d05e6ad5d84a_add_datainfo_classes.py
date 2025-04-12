"""[Add] DataInfo Classes

Revision ID: d05e6ad5d84a
Revises: 114a61cd9569
Create Date: 2025-04-12 12:09:13.810793

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd05e6ad5d84a'
down_revision: Union[str, None] = '114a61cd9569'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # MacroData_Info table
    op.create_table(
        'macro_data_info',
        sa.Column('data_name', sa.String(20), sa.ForeignKey('macro_data.data_name', name='macrodata_info_name_fkey'), nullable=False),
        sa.Column('country', sa.String(50), sa.ForeignKey('countries.name', name='macrodata_info_cntry_fkey'), nullable=False),
        sa.Column('last_update_date', sa.DateTime, nullable=True),
        sa.Column('description', sa.String(200), nullable=True),
        sa.Column('frequency', sa.String(20), nullable=True),
        sa.Column('source', sa.String(50), nullable=True),
        sa.PrimaryKeyConstraint('data_name', 'country', name='pk_macro_data_info'),
    )

    # MiscData_Info table
    op.create_table(
        'misc_data_info',
        sa.Column('data_name', sa.String(50), sa.ForeignKey('miscell_data.data_name', name='miscdata_info_name_fkey'), nullable=False, primary_key=True),
        sa.Column('last_update_date', sa.DateTime, nullable=True),
        sa.Column('description', sa.String(200), nullable=True),
        sa.Column('frequency', sa.String(20), nullable=True),
        sa.Column('source', sa.String(50), nullable=True),
    )


def downgrade() -> None:
    # Drop info tables
    op.drop_table('misc_data_info')
    op.drop_table('macro_data_info')
    