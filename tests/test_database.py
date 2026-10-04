"""ORM queries and deletion boundaries; requires the database dependencies."""

from datetime import datetime

import pytest


@pytest.fixture
def stored_data(monkeypatch):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    for dependency in ("dotenv", "yfinance", "investpy"):
        pytest.importorskip(dependency)
    from sqlalchemy.orm import sessionmaker

    from database import DB_Engine, models

    engine = sqlalchemy.create_engine("sqlite:///:memory:")

    @sqlalchemy.event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    models.Base.metadata.create_all(engine)
    monkeypatch.setattr(type(DB_Engine), "_instances", {})
    db = DB_Engine()
    db.engine = engine
    db.sessionmaker = sessionmaker(bind=engine)
    with db.sessionmaker() as session:
        session.add(models.Country("US", "US", "US", None, "USD", None))
        session.add_all([models.Company("A", "First", "US"), models.Company("B", "Other", "US")])
        session.flush()
        session.add_all(
            [
                models.PriceData(datetime(2020, 1, 2), "A", 11, 12),
                models.PriceData(datetime(2020, 1, 1), "A", 10, 11),
                models.PriceData(datetime(2020, 1, 1), "B", 20, 21),
                models.Pair(tickerA="A", tickerB="B", ceof=0.5),
            ]
        )
        session.commit()
    yield db, models
    engine.dispose()


def test_price_pair_and_country_queries_use_the_current_schema(stored_data):
    db, models = stored_data
    assert db.exists_pair(models.Pair(tickerA="A", tickerB="B", ceof=0.5))
    assert not db.exists_pair(models.Pair(tickerA="B", tickerB="A", ceof=0.5))
    assert db.exists_pricedata(models.PriceData(datetime(2020, 1, 1), "A", 99, 100))
    assert not db.exists_pricedata(models.PriceData(datetime(2020, 1, 3), "A", 10, 11))
    prices = db.get_all_pricedata("A", to_pandas=True)
    assert prices["close_price"].tolist() == [11, 12]
    assert prices["date"].is_monotonic_increasing
    assert db.get_all_pricedata("MISSING", to_pandas=True).empty
    country = models.Country("US", "US", "US", None, "USD", None)
    assert {company.ticker for company in db.get_all_country_companies(country)} == {"A", "B"}


def test_deletion_does_not_remove_unrelated_records(stored_data):
    db, models = stored_data
    with db.sessionmaker() as session:
        price = (
            session.query(models.PriceData).filter_by(ticker="A", date=datetime(2020, 1, 1)).one()
        )
        session.delete(price)
        session.commit()
        assert session.query(models.Company).count() == 2
        assert session.query(models.Country).count() == 1
        session.delete(session.query(models.Pair).one())
        session.commit()
        session.delete(session.query(models.Company).filter_by(ticker="A").one())
        session.commit()
        assert session.query(models.Country).count() == 1
        assert session.query(models.Asset).one().ticker == "B"
        assert session.query(models.PriceData).one().ticker == "B"
