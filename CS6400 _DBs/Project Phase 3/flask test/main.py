from flask import Flask, render_template, request, json, session, redirect, url_for
from flaskext.mysql import MySQL
from sql_utils import *
import pandas as pd

app = Flask(__name__)

mysql = MySQL()

# Create dictionary to store session vairables in
# session={}


app.secret_key = '_U1Ao7T+Awm!g,9v-HkS'

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'password123!'
app.config['MYSQL_DATABASE_DB'] = 'tradeplaza'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)

conn = mysql.connect()
cursor = conn.cursor()

@app.route('/acceptrejecttrade', methods=["GET", "POST"])
def acceptrejecttrade():
    counterparty_email = session['email']

    item_no_query = f"""
    SELECT t.proposer_item_no
    FROM Trade t
    where t.counterparty_email="{counterparty_email}"
    and status = "unaccepted"
    """
    cursor.execute(item_no_query)
    item_nos = cursor.fetchall()

    if item_nos:
        item_list = []
        for i in item_nos:
            item_list.append(int(str(i).replace("(", "").replace(")", "").replace(",", "")))
            if len(item_list) > 1:
                item_tuple = tuple(item_list)
            else:
                item_tuple = "(" + str(item_list[0]) + ")"

    proposed_trade_details_query = f"""
    SELECT i2.item_no as proposed_item, proposed_date, i1.title as desired_item, u.nickname as proposer, i2.title as proposed_item
    FROM Trade t
    join Item i1
    on t.counterparty_item_no=i1.item_no
    join Item i2
    on t.proposer_item_no=i2.item_no
    join User u
    on i2.lister_email=u.email
    where t.counterparty_email="{counterparty_email}"
    and status = "unaccepted"
    """

    cursor.execute(proposed_trade_details_query)
    td = cursor.fetchall()
    df_trade_details = pd.DataFrame(td, columns=['proposed_item_no', 'Date', 'Desired Item', 'Proposer', 'Proposed Item'])

    print(df_trade_details)

    proposer_rank_query = """
    SELECT item_no, rank_label FROM
    (SELECT rev_items.item_no, rev_items.lister_email, completed_trades FROM
    (SELECT item_no, lister_email from tradeplaza.Item where item_no in {})  rev_items
    LEFT JOIN
    (SELECT lister_email, count(distinct auto_trade_id) as completed_trades 
    FROM
    (SELECT lister_email, auto_trade_id FROM
    (SELECT lister_email, item_no from tradeplaza.Item where item_no in {}) item_sellers
    LEFT JOIN 
    tradeplaza.Trade t1
    on item_sellers.lister_email=t1.counterparty_email
    where t1.status<>"unaccepted"
    UNION
    SELECT lister_email,auto_trade_id FROM
    (SELECT lister_email, item_no from tradeplaza.Item where item_no in {})  item_sellers
    LEFT JOIN 
    tradeplaza.Trade t2
    on item_sellers.lister_email=t2.proposer_email
    where t2.status<>"unaccepted") all_trade_union
    GROUP BY lister_email) trade_count
    on rev_items.lister_email=trade_count.lister_email) item_email_tradecount
    LEFT JOIN
    rank_lookup r
    on item_email_tradecount.completed_trades*1>=r.trade_lower_range*1 AND item_email_tradecount.completed_trades*1<=r.trade_upper_range*1
    """.format(item_tuple, item_tuple, item_tuple)

    cursor.execute(proposer_rank_query)
    proposer_rank = cursor.fetchall()
    df_proposer_rank = pd.DataFrame(proposer_rank, columns=['proposed_item_no', 'Rank'])
    print(df_proposer_rank)

    proposer_email_postalcode_query = """
            SELECT item_no, round(distance*0.621371,2) as lister_distance
            FROM
            (
            SELECT loc2.item_no, (((acos(sin((loc2.lat*pi()/180)) * sin((loc1.lat*pi()/180)) + cos((loc2.lat*pi()/180)) * cos((loc1.lat*pi()/180)) * cos(((loc2.lng- loc1.lng) * pi()/180)))) * 180/pi()) * 60 * 1.1515 * 1.609344) as distance
            FROM 
            (select latitude as lat, longitude as lng, postal_code from Location_Lookup where postal_code in (select postal_code from User where email="{}")) loc1
            cross join
            (SELECT item_no, latitude as lat, longitude as lng, Location_Lookup.postal_code 
                from Item join User 
                on Item.lister_email=User.email
                join Location_Lookup
                on User.postal_code=Location_Lookup.postal_code
                where Item.item_no in {}) loc2
            ) dist
            """.format(session['email'], item_tuple)

    cursor.execute(proposer_email_postalcode_query)
    proposer_distance = cursor.fetchall()
    df_distance = pd.DataFrame(proposer_distance, columns=['proposed_item_no', 'Distance'])
    print(df_distance)

    df2 = pd.merge(df_trade_details, df_proposer_rank, on='proposed_item_no')
    df3 = pd.merge(df2, df_distance, on='proposed_item_no')
    print(df3)

    auto_trade_id = 3

    get_trade_details_query = f"""
    SELECT u.first_name, u.email 
    FROM Trade t join User u
    on t.proposer_email=u.email
     WHERE auto_trade_id={auto_trade_id}
    """
    cursor.execute(get_trade_details_query)
    first_name = cursor.fetchone()[0]
    cursor.execute(get_trade_details_query)
    email = cursor.fetchone()[1]
    trade = ""

    if request.method == "POST":
        one = request.form.get('accept_button')
        two = request.form.get('reject_button')
        if one is not None:
            trade = 'accepted'
        if two is not None:
            trade = 'rejected'

    item_no_query = f"""
    SELECT proposer_item_no, counterparty_item_no 
    FROM trade WHERE auto_trade_id = {auto_trade_id}
    """
    cursor.execute(item_no_query)
    proposed_item_no = cursor.fetchone()[0]
    cursor.execute(item_no_query)
    desired_item_no = cursor.fetchone()[1]
    #print(proposed_item_no, "\n", desired_item_no)
    #print("\n",auto_trade_id)

    if trade == 'accepted':
        update_trade_details_query_accepted = f"""
        UPDATE Trade SET status= "accepted" where auto_trade_id={auto_trade_id}
        """
        cursor.execute(update_trade_details_query_accepted)
        msg = "Your trade has been accepted"
        return render_template('acceptedtrade.html', proposed_item_no=proposed_item_no,
                               desired_item_no=desired_item_no, first_name=first_name, email=email, msg=msg)

    if trade == 'rejected':
        update_trade_details_query_rejected = f"""
        UPDATE Trade SET status= "rejected" where auto_trade_id={auto_trade_id}
        """
        cursor.execute(update_trade_details_query_rejected)
        conn.commit()
        msg = "Your trade has been rejected"
        return render_template('rejectedtrade.html', proposed_item_no=proposed_item_no,
                               desired_item_no=desired_item_no, first_name=first_name, email=email, msg=msg)




    return render_template('acceptrejecttrade.html', tables=[df3.to_html(classes='data', header="true")],
                                   search_results_list=df3_list, proposed_item_no=proposed_item_no,
                               desired_item_no=desired_item_no)
#TODO: *************** UPDATE DATE FORMAT***************************

if __name__ == "__main__":
    app.run(debug=True)