import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def read_orders(orders_file):
    """
    Parser orders into the form:
        Date      datetime64[ns]
        Symbol            object
        Order             object
        Shares             int32
    This is how the order book looks like:
        Date,Symbol,Order,Shares
        2011-01-10,AAPL,BUY,1500
        2011-01-10,AAPL,SELL,1500
    """
    orders = pd.read_csv(orders_file,
                         index_col=['Date'],
                         dtype='|str, str, str,  i4',
                         parse_dates=['Date'])
    orders.sort_values(by="Date", inplace=True)
    return orders


def get_order_book_info(orders):
    """Return start_date, end_date, and symbols (as a list)."""
    start_date = orders.index[0]
    end_date = orders.index[-1]
    symbols = sorted(list((set(orders.Symbol.tolist()))))
    return start_date, end_date, symbols


def get_portfolio_value(holding, prices):
    """Calculate the current portofolio value."""
    value = 0
    for ticker, shares in holding.items():
        if ticker == 'cash':
            value += shares
        else:
            value += shares * prices[ticker]
    return value


def handle_order(date, order, holding, prices, commission, impact):
    """Process the order."""
    symbol, order, shares = order
    if shares == 0 and order == "":
        return  # empty order
    if pd.isnull(shares):
        return  # shares is nan

    # Allow indicating buying and selling via shares. If shares is positive we
    # buy and if it is negative we sell.
    if shares > 0 and order == "":
        order = "BUY"
    elif shares < 0 and order == "":
        order = "SELL"
        shares = abs(shares)

    adj_closing_price = prices[symbol]
    cost = shares * adj_closing_price
    # Charge commission and deduct impact penalty
    holding['cash'] -= (commission + impact * adj_closing_price * shares)
    if order.upper() == "BUY":
        # print(f"Buy  {shares:6} of {symbol:4} on {date}")
        holding['cash'] -= cost
        holding[symbol] += shares
    elif order.upper() == "SELL":
        # print(f"Sell {shares:6} of {symbol:4} on {date}")
        holding['cash'] += cost
        holding[symbol] -= shares
    else:
        raise Exception("Unexpected order type.")


def compute_portvals(orders_file, start_val=1000000, commission=9.95, impact=0.005):
    if isinstance(orders_file, pd.DataFrame):
        orders = orders_file
    else:
        orders = read_orders(orders_file)

    start_date, end_date, symbols = get_order_book_info(orders)

    # Tickers in the orderbook over the date_range in the order book.
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices['Portval'] = pd.Series(0.0, index=prices.index)

    # A dictionary to keep track of the assets we are holding.
    holding = {s: 0 for s in symbols}
    holding['cash'] = start_val

    # Iterate over all trading days that are in the (inclusive) range of the
    # order book dates. This implicitly ignores orders placed on non-trading
    # days.
    for date, values in prices.iterrows():
        # Process orders for that day.
        for date, order in orders.loc[date:date].iterrows():
            handle_order(date, order, holding, values, commission, impact)
        # Compute portfolio value at the end of day.
        values['Portval'] = get_portfolio_value(holding, values)

    return prices[['Portval']]

def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()