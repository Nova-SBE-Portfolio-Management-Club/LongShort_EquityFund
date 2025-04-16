"""[Add] DataInfo Classes

Revision ID: 12d47eafc111
Revises: 114a61cd9569
Create Date: 2025-04-12 12:09:13.810793

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '12d47eafc111'
down_revision: Union[str, None] = '114a61cd9569'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # MacroData_Info table
    op.create_table(
        'macro_data_info',
        sa.Column('data_name', sa.String(20), nullable=False),
        sa.Column('country', sa.String(50), sa.ForeignKey('countries.name', name='macrodatainfo_cntry_fkey', ondelete="CASCADE"), nullable=False),
        sa.Column('last_update_date', sa.DateTime, nullable=True),
        sa.Column('description', sa.String(200), nullable=True),
        sa.Column('frequency', sa.String(20), nullable=True),
        sa.Column('source', sa.String(50), nullable=True),
        sa.PrimaryKeyConstraint('data_name', 'country', name='pk_macro_data_info'),
    )

    # MacroData table
    op.create_table(
        'macro_data',
        sa.Column('country', sa.String(50), sa.ForeignKey('countries.name', name='macrodata_cntry_fkey', ondelete="CASCADE"), nullable=False),
        sa.Column('data_name', sa.String(20), nullable=False),
        sa.Column('date', sa.DateTime, nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('country', 'data_name', 'date', name='pk_macro_data'),
        sa.ForeignKeyConstraint(
            ['data_name', 'country'],
            ['macro_data_info.data_name', 'macro_data_info.country'],
            name='macrodata_info_fk',
            ondelete='CASCADE'
        )
    )

    # MiscData_Info table
    op.create_table(
        'misc_data_info',
        sa.Column('data_name', sa.String(50), primary_key=True, nullable=False),
        sa.Column('last_update_date', sa.DateTime, nullable=True),
        sa.Column('description', sa.String(200), nullable=True),
        sa.Column('frequency', sa.String(20), nullable=True),
        sa.Column('source', sa.String(50), nullable=True),
    )

    # MiscellData table
    op.create_table(
        'miscell_data',
        sa.Column('data_name', sa.String(50), sa.ForeignKey('misc_data_info.data_name', name='miscdata_name_fkey'), nullable=False),
        sa.Column('date', sa.DateTime, nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('data_name', 'date', name='pk_miscell_data'),
    )


def downgrade() -> None:
    op.drop_table('miscell_data')
    op.drop_table('misc_data_info')
    op.drop_table('macro_data')
    op.drop_table('macro_data_info')