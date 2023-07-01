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


@app.route('/')
def index():
    if 'email' in session:
        return f'Logged in as {session["email"]}'
    return render_template('login.html')


@app.route("/login")
def main():
    return render_template('login.html')


@app.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('email', None)
    return redirect(url_for('index'))


@app.route('/api/login', methods=['POST'])
def login():
    msg = "Welcome to Trade Plaza"
    _username = request.form['username']
    _password = request.form['password']

    cursor.execute('SELECT * FROM User WHERE nickname = %(username)s OR email=%(username)s ', {'username': _username})
    account_exists = cursor.fetchone()
    print(account_exists)

    if not account_exists:
        msg = 'That account does not exist'
        return render_template('login.html', msg=msg)

    if account_exists:
        cursor.execute('SELECT password,email FROM User WHERE nickname = %(username)s OR email=%(username)s ',
                       {'username': _username})
        row = cursor.fetchone()
        correct_password = row[0]
        session_email = row[1]
        if _password == correct_password:
            session['email'] = session_email
            session.modified = True
            cursor.execute('SELECT first_name,last_name,nickname from User where email= %s', (session["email"]))
            row = cursor.fetchone()
            first_name = row[0]
            last_name = row[1]
            nickname = row[2]

            ##unaccepted trades
            cursor.execute('select coalesce(count(*),0) from trade where status="unaccepted" and counterparty_email=%s',
                           (session['email']), )
            unaccepted_trades = cursor.fetchone()[0]

            ##average response time
            cursor.execute(
                'SELECT round(coalesce(avg(datediff(accept_reject_date,proposed_date)),1),"None") FROM trade WHERE counterparty_email= %s and status<>"unaccepted" ',
                (session['email']), )
            avg_response = cursor.fetchone()[0]

            ##average response time color
            text_color = get_text_color_from_response_time(cursor, conn, avg_response)

            ##completed trades
            cursor.execute(
                'select coalesce(count(*),0) from trade where status<>"unaccepted" and (counterparty_email=%s or proposer_email=%s)',
                (session['email'], session['email']))
            completed_trade_count = cursor.fetchone()[0]

            ##user_rank
            cursor.execute('SELECT rank_label from rank_lookup WHERE trade_lower_range<=%s and trade_upper_range>=%s',
                           (completed_trade_count, completed_trade_count))
            user_rank = cursor.fetchone()[0]

            ##update main menu page
            return render_template('mainmenu.html', msg=msg, first_name=first_name, last_name=last_name,
                                   nickname=nickname, avg_response=avg_response, unaccepted_trades=unaccepted_trades,
                                   user_rank=user_rank, text_color=text_color)
        else:
            msg = 'Incorrect password'
            return render_template('login.html', msg=msg)


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/api/register', methods=['POST'])
def Register():
    # read the posted values from the UI
    _email = request.form['email']
    _password = request.form['password']
    _nickname = request.form['nickname']
    _firstname = request.form['firstname']
    _lastname = request.form['lastname']
    _postalcode = request.form['postalcode']

    # validate the received values
    if _email and _password and _nickname and _firstname and _lastname and _postalcode:
        msg = 'Success'

        cursor.execute('SELECT email FROM User WHERE email = %s', (_email,))
        account_email = cursor.fetchone()

        if account_email:
            msg = 'Email already in use!'

        cursor.execute('SELECT email FROM User WHERE nickname = %s', (_nickname,))
        account_nickname = cursor.fetchone()

        if account_nickname:
            msg = 'That nickname is taken!'

        cursor.execute('SELECT postal_code FROM Location_Lookup WHERE postal_code= %s', (_postalcode,))
        postalcode = cursor.fetchone()

        if not postalcode:
            msg = 'That is not a known postal code'

        if not account_email and not account_nickname and postalcode:
            cursor.execute(
                'INSERT INTO User (email,password,nickname,first_name,last_name,postal_code) VALUES (%s, %s, %s, %s, %s, %s)',
                (_email, _password, _nickname, _firstname, _lastname, _postalcode))
            conn.commit()
            return render_template('mainmenu.html', msg=msg, first_name=_firstname, last_name=_lastname,
                                   nickname=_nickname, avg_response=avg_response)


    else:
        msg = 'Please enter all required fields'

    return render_template('register.html', msg=msg)


@app.route('/mainmenu')
def mainmenu():
    if 'email' in session:
        return render_template('mainmenu.html')
    else:
        return render_template('login.html')


@app.route('/listitem')
def listitem():
    cursor.execute('SELECT name, friendly_platform_name FROM platform')
    platform_vg_list = cursor.fetchall()
    return render_template('listitem.html', platform_vg_list=platform_vg_list)


@app.route('/api/listitem', methods=['POST'])
def ListItem():
    # read the posted values from the UI
    _game_type = request.form['game_type']
    _game_title = request.form['game_title']
    _condition = request.form['condition']
    _description = request.form['description']
    _cardno = request.form['no_cards']
    _media = request.form['media']
    _platform_vg = request.form['platform_vg']
    _platform_cg = request.form['platform_cg']

    # validate the received values
    if (_game_type == "collectable_card_game" and _cardno.isdigit() == False):
        msg = 'You must enter a round number of cards (e.g. 4, 27, not 3.5)'
        return render_template('listitem.html', msg=msg)

    if not _game_type or not _game_title or not _condition or not _description or (
            _game_type == "collectable_card_game" and not _cardno):
        msg = 'Please fill out all required fields'
        return render_template('listitem.html', msg=msg)

    if _game_type and _game_title and _condition and _description:

        # because we need to insert into two tables, we can't autoincrement
        # find the highest item_no and add 1 to it
        cursor.execute('SELECT COALESCE(max(item_no),1)+1 FROM Item')
        row = cursor.fetchone()
        current_item_no = row[0]

        cursor.execute(
            'INSERT INTO Item (lister_email,title,item_no,`condition`,description,listing_url) VALUES (%s,%s,%s,%s,%s,%s)',
            (session['email'], _game_title, current_item_no, _condition, _description, "sample.com"))
        conn.commit()

        if _game_type == 'board_game':
            cursor.execute('INSERT INTO Item_Board_Game (lister_email,item_no) VALUES (%s,%s)',
                           (session['email'], current_item_no))
            conn.commit()

        if _game_type == 'playing_card_game':
            cursor.execute('INSERT INTO Item_Playing_Card_Game (lister_email,item_no) VALUES (%s,%s)',
                           (session['email'], current_item_no))
            conn.commit()

        if _game_type == 'collectable_card_game':
            cursor.execute(
                'INSERT INTO Item_Collectable_Card_Game (lister_email,item_no,number_of_cards) VALUES (%s,%s,%s)',
                (session['email'], current_item_no, _cardno))
            conn.commit()

        if _game_type == 'computer_game':
            cursor.execute('INSERT INTO Item_Computer_Game (lister_email,item_no,platform) VALUES (%s,%s,%s)',
                           (session['email'], current_item_no, _platform_cg))
            conn.commit()

        if _game_type == 'video_game':
            cursor.execute('INSERT INTO Item_Video_Game (lister_email,item_no,platform,media) VALUES (%s,%s,%s,%s)',
                           (session['email'], current_item_no, _platform_vg, _media))
            conn.commit()

        return render_template('itemconfirmation.html', current_item_no=current_item_no)


@app.route('/myitems')
def myitems():
    cursor.execute(
        'SELECT count(*) FROM Item i RIGHT JOIN Item_Board_Game bg on i.item_no=bg.item_no WHERE i.lister_email=%s and i.item_no not in (select proposer_item_no from Trade where status= "accepted") and i.item_no not in (select counterparty_item_no from Trade WHERE status= "accepted")',
        (session["email"]))
    boardgamecount = cursor.fetchone()[0]
    cursor.execute(
        'SELECT count(*) FROM Item i RIGHT JOIN Item_Playing_Card_Game pcg on i.item_no=pcg.item_no WHERE i.lister_email=%s and i.item_no not in (select proposer_item_no from Trade where status= "accepted") and i.item_no not in (select counterparty_item_no from Trade WHERE status= "accepted")',
        (session["email"]))
    playcardcount = cursor.fetchone()[0]
    cursor.execute(
        'SELECT count(*) FROM Item i RIGHT JOIN Item_Computer_Game pcg on i.item_no=pcg.item_no WHERE i.lister_email=%s and i.item_no not in (select proposer_item_no from Trade where status= "accepted") and i.item_no not in (select counterparty_item_no from Trade WHERE status= "accepted")',
        (session["email"]))
    computergamecount = cursor.fetchone()[0]
    cursor.execute(
        'SELECT count(*) FROM Item i RIGHT JOIN Item_Collectable_Card_Game pcg on i.item_no=pcg.item_no WHERE i.lister_email=%s and i.item_no not in (select proposer_item_no from Trade where status= "accepted") and i.item_no not in (select counterparty_item_no from Trade WHERE status= "accepted")',
        (session["email"]))
    collectcardgamecount = cursor.fetchone()[0]
    cursor.execute(
        'SELECT count(*) FROM Item i RIGHT JOIN Item_Video_Game pcg on i.item_no=pcg.item_no WHERE i.lister_email=%s and i.item_no not in (select proposer_item_no from Trade where status= "accepted") and i.item_no not in (select counterparty_item_no from Trade WHERE status= "accepted")',
        (session["email"]))
    videogamecount = cursor.fetchone()[0]
    cursor.execute(
        'SELECT count(*) FROM Item i WHERE i.lister_email=%s and i.item_no not in (select proposer_item_no from Trade where status= "accepted") and i.item_no not in (select counterparty_item_no from Trade WHERE status= "accepted")',
        (session["email"]))
    totalcount = cursor.fetchone()[0]

    get_item_details = """
    SELECT i.item_no, 
    CASE WHEN cg.item_no IS NOT NULL then "Collectable Card Game" 
    WHEN bg.item_no IS NOT NULL then "Board Game" 
    WHEN pcg.item_no IS NOT NULL then "Playing Card Game" 
    WHEN comg.item_no is NOT NULL then "Computer Game" 
    WHEN vg.item_no is NOT NULL then "Video Game" 
    END as game_type, title, `condition`, description 
    FROM Item i 
    LEFT JOIN Item_Collectable_Card_Game cg on i.item_no=cg.item_no 
    LEFT JOIN Item_Board_Game bg on i.item_no=bg.item_no 
    LEFT JOIN Item_Playing_Card_Game pcg on i.item_no=pcg.item_no 
    LEFT JOIN Item_Computer_Game comg on i.item_no=comg.item_no 
    LEFT JOIN Item_Video_Game vg on i.item_no=vg.item_no 
    WHERE i.lister_email=%s and i.item_no not in 
    (select proposer_item_no from Trade where status= "accepted") 
    and i.item_no not in (select counterparty_item_no from Trade where status= "accepted")
    """
    cursor.execute(get_item_details, (session["email"]))
    data = cursor.fetchall()
    return render_template('myitems.html', boardgamecount=boardgamecount, playcardcount=playcardcount,
                           computergamecount=computergamecount, collectcardgamecount=collectcardgamecount,
                           videogamecount=videogamecount, totalcount=totalcount, data=data)


@app.route('/searchitems')
def searchitems():
    return render_template('searchitems.html')


@app.route('/displaysearch')
def displaysearch():
    return render_template('displaysearch.html')


@app.route('/api/search', methods=['POST'])
def SearchItems():
    _searchtype = request.form['searchtype']
    _search_postal_code = request.form['postalcode_search']
    _search_keyword = request.form['keyword_search'].lower()
    _mile_search = request.form['mile_search']

    no_results_found = "Sorry no results found!"

    if _searchtype == "by_keyword":
        cursor.execute('''SELECT item_no 
                        FROM Item 
                        WHERE 
                        lister_email<> %s
                        AND (lower(description) like %s or lower(title) like %s )
                        AND (item_no not in (select distinct proposer_item_no from Trade where status= 'accepted') 
                        AND item_no not in (select distinct counterparty_item_no from Trade where status= 'accepted'))
                        ''', (session['email'], '%' + _search_keyword + '%', '%' + _search_keyword + '%'))

        item_nos = cursor.fetchall()

    if _searchtype == "by_my_postalcode":
        cursor.execute('SELECT postal_code FROM User WHERE email = %s', (session['email']))
        user_postal_code = cursor.fetchone()

        cursor.execute('''SELECT item_no 
                        FROM Item 
                        WHERE lister_email in (SELECT email from User where postal_code= %s and email<> %s) 
                        AND (item_no not in (select distinct proposer_item_no from Trade where status= 'accepted') 
                        AND item_no not in (select distinct counterparty_item_no from Trade where status= 'accepted'))
                        ''', (user_postal_code, session['email']))

        item_nos = cursor.fetchall()

    if _searchtype == "within_xmiles":
        cursor.execute('''
            SELECT item_no from Item 
            where lister_email in 
            (SELECT email from User where postal_code in (
            SELECT postal_code
            FROM
            (
            SELECT loc2.postal_code, (((acos(sin((loc2.lat*pi()/180)) * sin((loc1.lat*pi()/180)) + cos((loc2.lat*pi()/180)) * cos((loc1.lat*pi()/180)) * cos(((loc2.lng- loc1.lng) * pi()/180)))) * 180/pi()) * 60 * 1.1515 * 1.609344) as distance
            FROM 
            (select latitude as lat, longitude as lng, postal_code from Location_Lookup where postal_code in (select postal_code from User where email= %s)) loc1
            cross join
            (SELECT latitude as lat, longitude as lng, postal_code from Location_Lookup) loc2
            ) dist
            WHERE distance*0.621371 <= %s
            ) and email<>%s) and 
             (item_no not in (select distinct proposer_item_no from Trade where status= 'accepted') 
            AND item_no not in (select distinct counterparty_item_no from Trade where status= 'accepted'))
            ''', (session['email'], _mile_search, session['email']))

        item_nos = cursor.fetchall()

    if _searchtype == "by_postalcode":
        cursor.execute('''SELECT item_no 
                        FROM Item 
                        WHERE lister_email in (SELECT email from User where postal_code= %s and email<> %s) 
                        AND (item_no not in (select distinct proposer_item_no from Trade where status= 'accepted') 
                        AND item_no not in (select distinct counterparty_item_no from Trade where status= 'accepted'))
                        ''', (_search_postal_code, session['email']))

        item_nos = cursor.fetchall()

    if item_nos:
        item_list = []

        for i in item_nos:
            item_list.append(int(str(i).replace("(", "").replace(")", "").replace(",", "")))

        item_detail_query = """
        SELECT i.item_no, title
            , CASE WHEN cg.item_no IS NOT NULL then 'Collectable Card Game'
            WHEN bg.item_no IS NOT NULL then 'Board Game'
            WHEN pcg.item_no IS NOT NULL then 'Playing Card Game'
            WHEN comg.item_no is NOT NULL then 'Computer Game'
            WHEN vg.item_no is NOT NULL then 'Video Game' END as game_type
            , `condition`, listing_url
            , CASE WHEN LENGTH(description)<=100 THEN description
             WHEN LENGTH(description)>100 THEN CONCAT(LEFT(description,100),"...")
             END as description
            FROM Item i
            LEFT JOIN Item_Collectable_Card_Game cg on i.item_no=cg.item_no
            LEFT JOIN Item_Board_Game bg on
            i.item_no=bg.item_no
            LEFT JOIN Item_Playing_Card_Game pcg on i.item_no=pcg.item_no
            LEFT JOIN Item_Computer_Game comg on i.item_no=comg.item_no
            LEFT JOIN Item_Video_Game vg on i.item_no=vg.item_no
            WHERE i.item_no in {}
            """.format(tuple(item_list))

        cursor.execute(item_detail_query)
        items = cursor.fetchall()
        items = pd.DataFrame(items,
                             columns=['Item Number', 'Title', 'Game type', 'Condition', 'Listing URL', 'Description'])

        lister_response_query = """
        SELECT rev_items.item_no, COALESCE(lister_avg_response_time,"None") as lister_avg_response_time 
        FROM
        (SELECT item_no, lister_email from tradeplaza.Item where item_no in {}) rev_items
        LEFT JOIN
        (SELECT item_sellers.lister_email
        ,round(avg(datediff(accept_reject_date,proposed_date)),1) as lister_avg_response_time
        FROM
        (SELECT lister_email, item_no from tradeplaza.Item where item_no in {}) item_sellers
        LEFT JOIN 
        tradeplaza.Trade t
        on item_sellers.lister_email=t.counterparty_email
        WHERE (t.status= 'accepted' or t.status='rejected')
        GROUP BY lister_email) response_times
        on rev_items.lister_email=response_times.lister_email
        """.format(tuple(item_list), tuple(item_list))

        cursor.execute(lister_response_query)
        lister_response = cursor.fetchall()
        lister_response = pd.DataFrame(lister_response, columns=['Item Number', 'Response Time (Days)'])

        lister_rank_query = """
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
        """.format(tuple(item_list), tuple(item_list), tuple(item_list))

        cursor.execute(lister_rank_query)
        lister_rank = cursor.fetchall()
        lister_rank = pd.DataFrame(lister_rank, columns=['Item Number', 'Rank'])
        print(lister_rank)

        lister_distance_query = """
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
        """.format(session['email'], tuple(item_list))

        cursor.execute(lister_distance_query)
        lister_distance = cursor.fetchall()
        lister_distance = pd.DataFrame(lister_distance, columns=['Item Number', 'Distance'])

        df2 = pd.merge(items, lister_response, on='Item Number')
        df3 = pd.merge(df2, lister_rank, on='Item Number')
        search_results = pd.merge(df3, lister_distance, on='Item Number')

        # Also save the list version to HTML page, so we can insert link to the table cell
        search_results_list = search_results.values.tolist()
        msg = "Search Results"

        if _searchtype == "by_keyword":
            # TODO: the keyword is not yet highlighted in the HTML table
            return render_template('displaysearch.html', keyword=_search_keyword, msg=msg,
                                   tables=[search_results.to_html(classes='data', header="true")],
                                   search_results_list=search_results_list)
        else:
            return render_template('displaysearch.html', msg=msg,
                                   tables=[search_results.to_html(classes='data', header="true")],
                                   search_results_list=search_results_list)

    else:
        msg = no_results_found
        return render_template('searchitems.html', msg=msg)


@app.route('/tradehistory')
def tradehistory():
    query = "proposer_email+cagim3@gatech.edu+counterparty_email+ebentivegna3@gatech.edu+proposer_item_no+7+counterparty_item_no+8"
    return render_template('tradehistory.html', query=query)


@app.route('/tradedetails/<path:path>')
def tradedetails(path):
    lst = path.split("+")
    list = iter(lst)
    dct = dict(zip(list, list))
    proposer_email = dct["proposer_email"]
    counterparty_email = dct["counterparty_email"]
    proposer_item_no = dct["proposer_item_no"]
    counterparty_item_no = dct["counterparty_item_no"]

    select_proposed_date = f"""
    SELECT proposed_date 
    FROM Trade 
    WHERE proposer_email="{proposer_email}" and counterparty_email="{counterparty_email}" and proposer_item_no={proposer_item_no} and counterparty_item_no={counterparty_item_no}
    """
    cursor.execute(select_proposed_date)
    proposed_date = cursor.fetchone()[0]

    select_accept_reject_date = f"""
    SELECT accept_reject_date 
    FROM Trade 
    WHERE proposer_email="{proposer_email}" and counterparty_email="{counterparty_email}" and proposer_item_no={proposer_item_no} and counterparty_item_no={counterparty_item_no}
    """
    cursor.execute(select_accept_reject_date)
    accept_reject_date = cursor.fetchone()[0]

    select_status = f"""
    SELECT status 
    FROM Trade 
    WHERE proposer_email="{proposer_email}" and counterparty_email="{counterparty_email}" and proposer_item_no={proposer_item_no} and counterparty_item_no={counterparty_item_no}
    """
    cursor.execute(select_status)
    status = cursor.fetchone()[0]

    if proposer_email == (session['email']):
        my_role = "Proposer"
    else:
        my_role = "Counterparty"

    response_time = (accept_reject_date - proposed_date).days

    if my_role == "Proposer":
        select_other_nickname = f"""
        SELECT nickname 
        FROM User 
        WHERE email="{counterparty_email}"
        """
        cursor.execute(select_other_nickname)
        other_user_nickname = cursor.fetchone()[0]
    else:
        select_other_nickname = f"""
        SELECT nickname 
        FROM User 
        WHERE email="{proposer_email}"
        """
        cursor.execute(select_other_nickname)
        other_user_nickname = cursor.fetchone()[0]

    # TODO: DISTANCE (and adding back listing_url)

    if my_role == "Proposer":
        select_other_name = f"""
        SELECT first_name 
        FROM User 
        WHERE email="{counterparty_email}"
        """
        cursor.execute(select_other_name)
        other_user_name = cursor.fetchone()[0]
    else:
        select_other_name = f"""
        SELECT first_name 
        FROM User 
        WHERE email="{proposer_email}"
        """
        cursor.execute(select_other_name)
        other_user_name = cursor.fetchone()[0]

    if my_role == "Proposer":
        select_other_email = f"""
        SELECT email 
        FROM User 
        WHERE email="{counterparty_email}"
        """
        cursor.execute(select_other_email)
        other_user_email = cursor.fetchone()[0]
    else:
        select_other_email = f"""
        SELECT first_name 
        FROM User 
        WHERE email="{proposer_email}"
        """
        cursor.execute(select_other_email)
        other_user_email = cursor.fetchone()[0]

    select_proposed_item_no = f"""
    SELECT proposer_item_no 
    FROM Trade 
    WHERE proposer_email="{proposer_email}" and counterparty_email="{counterparty_email}" and proposer_item_no={proposer_item_no} and counterparty_item_no={counterparty_item_no}
    """
    cursor.execute(select_proposed_item_no)
    proposed_item_no = cursor.fetchone()[0]

    # TODO: title and gametype

    select_proposed_condition = f"""
    SELECT `condition`
    FROM Item
    WHERE item_no={proposer_item_no}
    """
    cursor.execute(select_proposed_condition)
    proposed_condition = cursor.fetchone()[0]

    # TODO: logic for description over 100 characters
    select_proposed_description = f"""
    SELECT description
    FROM Item
    WHERE item_no={proposed_item_no}
    """
    cursor.execute(select_proposed_description)
    proposed_description = cursor.fetchone()[0]

    select_desired_item_no = f"""
    SELECT counterparty_item_no 
    FROM Trade 
    WHERE proposer_email="{proposer_email}" and counterparty_email="{counterparty_email}" and proposer_item_no={proposer_item_no} and counterparty_item_no={counterparty_item_no}
    """
    cursor.execute(select_desired_item_no)
    desired_item_no = cursor.fetchone()[0]

    select_desired_condition = f"""
    SELECT `condition`
    FROM Item
    WHERE item_no={counterparty_item_no}
    """
    cursor.execute(select_desired_condition)
    desired_condition = cursor.fetchone()[0]

    return render_template('tradedetails.html', proposed_date=proposed_date.strftime("%x"),
                           accept_reject_date=accept_reject_date.strftime("%x"), status=status, my_role=my_role,
                           response_time=response_time, other_user_nickname=other_user_nickname,
                           other_user_name=other_user_name, other_user_email=other_user_email,
                           proposed_item_no=proposed_item_no, proposed_condition=proposed_condition,
                           proposed_description=proposed_description, desired_item_no=desired_item_no,
                           desired_condition=desired_condition)


@app.route('/proposetrade')
def proposetrade():
    return ""


@app.route('/acceptrejecttrade', methods=["GET", "POST"])
def acceptrejecttrade():
    #counterparty_email = session["email"]
    counterparty_email = "usr294@gt.edu"

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
    print("item list ", item_list)
    count = len(item_list)


    proposed_trade_details_query = f"""
    SELECT i2.item_no as proposed_item, t.counterparty_item_no, proposed_date, i1.title as desired_item, u.nickname as proposer, i2.title as proposed_item, t.auto_trade_id, u.first_name, u.last_name, u.email
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
    df_trade_details = pd.DataFrame(td, columns=['proposed_item_no', 'counterparty_item_no', 'Date', 'Desired Item', 'Proposer', 'Proposed Item', 'auto_trade_id', 'firstname','lastname', 'email'])
    print("df trade details", df_trade_details)

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
    print("df2", df2)
    df3 = pd.merge(df2, df_distance, on='proposed_item_no')
    print("df3", df3)
    df3_list = df3.values.tolist()
    print("column index")
    print(df3.columns.get_loc("Distance"))

    counter_party_item_number = df3['counterparty_item_no'].tolist()
    proposer_item_number = df3['proposed_item_no'].tolist()


    trade = ""
    if request.method == "POST":
        one = request.form.get('accept_button')
        two = request.form.get('reject_button')
        if one is not None:
            trade = 'accepted'
        if two is not None:
            trade = 'rejected'
        if count != 0:
            if trade == 'accepted':
                cursor.execute(f"""
                UPDATE Trade SET status= "accepted" where (counterparty_item_no = {counter_party_item_number[0]} AND proposer_item_no= {proposer_item_number[0]})
                """)
                conn.commit()
                msg = "Your trade has been accepted"
                count = count-1
                return render_template('acceptedtrade.html', tables=[df3.to_html(classes='data', header="true")],
                                           search_results_list=df3_list, msg=msg)

            if trade == 'rejected':
                cursor.execute(f"""
                UPDATE Trade SET status= "rejected" where (counterparty_item_no = {counter_party_item_number[0]} AND proposer_item_no= {proposer_item_number[0]})
                """)
                conn.commit()
                msg = "This trade has been rejected"
                count = count - 1
                return render_template('rejectedtrade.html', tables=[df3.to_html(classes='data', header="true")],
                                           search_results_list=df3_list, msg=msg)
        else:
            return render_template('mainmenu.html')

    return render_template('acceptrejecttrade.html', tables=[df3.to_html(classes='data', header="true")],
                                   search_results_list=df3_list)
#TODO: *************** UPDATE DATE FORMAT***************************


@app.route("/itemdetails/<path:path>", methods=['GET', 'POST'])
def itemdetails(path):
    '''
    There are two ways to redirect to this page:
    - (1) Go to "My Items", and click "Details"
    - (2) Go to "Search Items", and click "Details"
    There are 2 parts that will be shown here:
    - (1) Item Info
    - (2) Item Owner Info (Optional, only when account_owner != item_owner)
    '''

    # >>> Variables the will be passed to HTML page in the end of the function to render the item details.
    is_my_own_item = False
    item_owner_first_name = ""
    item_owner_last_name = ""
    item_onwer_postal_code = ""
    item_owner_city = ""
    item_owner_state = ""
    distance = 0
    background_color = ""
    video_game_platform = ""
    media = ""
    computer_platform = ""
    is_video_game = False
    is_collectible_card_game = False
    is_playing_card_game = False
    is_board_game = False
    is_computer_game = False
    number_of_cards = 0

    # >>> process the selected item
    item_no = path
    print("[Debug/itemdetails] =====================================================")
    print("[Debug/itemdetails] You are checking the details of this item_no: ", item_no)
    print("[Debug/itemdetails] =====================================================")

    # >>> Interact with SQL DataBase
    cmd = """SELECT * FROM Item WHERE item_no = {};""".format(item_no)
    # print("[Debug/itemdetails] about to run cmd: ", cmd)
    cursor.execute(cmd)
    item_row = cursor.fetchone()
    print("[Debug/itemdetails] The item you're looking for: ", item_row)
    lister_email = item_row[0]
    title = item_row[1]
    item_no = item_row[2]
    condition = item_row[3]
    description = item_row[4]
    # listing_url = item_row[5]
    # print("[Debug/itemdetails] lister_email: ", lister_email)
    # print("[Debug/itemdetails] title: ", title)
    # print("[Debug/itemdetails] item_no: ", item_no)
    # print("[Debug/itemdetails] condition: ", condition)
    # print("[Debug/itemdetails] description: ", description)
    # print("[Debug/itemdetails] listing_url: ", listing_url)

    # >>> check if Item is VIDEO GAME (if yes, get PLATFORM and MEDIA info)
    cmd = "select exists(select * from Item_Video_Game where item_no = {})".format(item_no)
    # print("[Debug/itemdetails] about to run cmd: ", cmd)
    cursor.execute(cmd)
    item_row = cursor.fetchone()
    is_video_game = item_row[0]
    if (is_video_game):
        # print("[Debug/itemdetails] is_video_game: ", is_video_game)
        cmd = "select * from Item_Video_Game where item_no = {}".format(item_no)
        # print("[Debug/itemdetails] about to run cmd: ", cmd)
        cursor.execute(cmd)
        item_row = cursor.fetchone()
        video_game_platform = item_row[2]
        media = item_row[3]
        # print("[Debug/itemdetails] platform: ", video_game_platform)
        # print("[Debug/itemdetails] media: ", media)

    # >>> check if Item is COLLECTIBLE CARD GAME (if yes, get NUMBER_OF_CARDS info)
    cmd = "select exists(select * from Item_Collectable_Card_Game where item_no = {})".format(item_no)
    # print("[Debug/itemdetails] about to run cmd: ", cmd)
    cursor.execute(cmd)
    item_row = cursor.fetchone()
    is_collectible_card_game = item_row[0]
    if (is_collectible_card_game):
        # print("[Debug/itemdetails] is_video_game: ", is_video_game)
        cmd = "select * from Item_Collectable_Card_Game where item_no = {}".format(item_no)
        # print("[Debug/itemdetails] about to run cmd: ", cmd)
        cursor.execute(cmd)
        item_row = cursor.fetchone()
        number_of_cards = item_row[2]
        # print("[Debug/itemdetails] number_of_cards: ", number_of_cards)

    # >>> check if Item is PLAYING CARD GAME (if yes, no further info is needed)
    cmd = "select exists(select * from Item_Playing_Card_Game where item_no = {})".format(item_no)
    print("[Debug/itemdetails] about to run cmd: ", cmd)
    cursor.execute(cmd)
    item_row = cursor.fetchone()
    is_playing_card_game = item_row[0]
    if (is_playing_card_game):
        print("[Debug/itemdetails] is_playing_card_game: ", is_playing_card_game)

    # >>> check if Item is BOARD GAME (if yes, no further info is needed)
    cmd = "select exists(select * from Item_Board_Game where item_no = {})".format(item_no)
    # print("[Debug/itemdetails] about to run cmd: ", cmd)
    cursor.execute(cmd)
    item_row = cursor.fetchone()
    is_board_game = item_row[0]
    if (is_board_game):
        print("[Debug/itemdetails] is_board_game: ", is_board_game)

    # >>> check if Item is COMPUTER GAME (if yes, get PLATFORM info)
    cmd = "select exists(select * from Item_Computer_Game where item_no = {})".format(item_no)
    # print("[Debug/itemdetails] about to run cmd: ", cmd)
    cursor.execute(cmd)
    item_row = cursor.fetchone()
    is_computer_game = item_row[0]
    if (is_computer_game):
        # print("[Debug/itemdetails] is_computer_game: ", is_computer_game)
        cmd = "select * from Item_Computer_Game where item_no = {}".format(item_no)
        # print("[Debug/itemdetails] about to run cmd: ", cmd)
        cursor.execute(cmd)
        item_row = cursor.fetchone()
        computer_platform = item_row[2]
        # print("[Debug/itemdetails] computer_platform: ", computer_platform)

    # >>> check ownership. show more info if you are viewing other's item
    if (session['email'] != lister_email):  # other's item
        print("[Debug/itemdetails] You are viewing other person's item")
        is_my_own_item = False

        # >>> get the item owner info from SQL
        cmd = """SELECT * FROM User WHERE email = '{}';""".format(lister_email)
        # print("[Debug/itemdetails] about to run cmd: ", cmd)
        cursor.execute(cmd)
        row = cursor.fetchone()
        # print("[Debug/itemdetails] item owner info: ", row)
        item_owner_first_name = row[2]
        item_owner_last_name = row[3]
        item_onwer_postal_code = row[5]
        # print("[Debug/itemdetails] item_owner_first_name: ", item_owner_first_name)
        # print("[Debug/itemdetails] item_owner_last_name: ", item_owner_last_name)
        # print("[Debug/itemdetails] item_onwer_postal_code: ", item_onwer_postal_code)

        # >>> get item owner's postal code
        cmd = """SELECT * FROM Location_Lookup WHERE postal_code = '{}';""".format(item_onwer_postal_code)
        # print("[Debug/itemdetails] about to run cmd: ", cmd)
        cursor.execute(cmd)
        row = cursor.fetchone()
        print("[Debug/itemdetails] item_onwer_postal_code: ", row)
        item_owner_city = row[1]
        item_owner_state = row[2]
        # item_owner_latitude = row[3]
        # item_owner_longitude = row[4]

        # >>> get account owner's postal code
        cmd = """SELECT postal_code FROM User WHERE email = '{}';""".format(session['email'])
        # print("[Debug/itemdetails] about to run cmd: ", cmd)
        cursor.execute(cmd)
        row = cursor.fetchone()
        account_owner_postal_code = row[0]
        print("[Debug/itemdetails] account_owner_postal_code: ", account_owner_postal_code)

        # >>> Calculate the distance between the account_owner and item_owner
        cmd = """SELECT (((acos(sin((lat2*pi()/180)) * sin((lat1*pi()/180)) + cos((lat2*pi()/180)) * cos((lat1*pi()/180)) * cos(((lon2- lon1) * pi()/180)))) * 180/pi()) * 60 * 1.1515 * 1.609344) as distance 
                FROM((SELECT latitude as lat1, longitude as lon1 FROM Location_Lookup WHERE postal_code = '{}') t1 
                JOIN 
                (SELECT latitude as lat2, longitude as lon2 FROM Location_Lookup WHERE postal_code = '{}') t2);
            """.format(account_owner_postal_code, item_onwer_postal_code)
        # print("[Debug/itemdetails] about to run cmd: ", cmd)
        cursor.execute(cmd)
        row = cursor.fetchone()
        distance = row[0]

        # >>> calculate the distance TODO: the distance cannot be calculated this way
        # distance = calculate_distance_between_postal_code(account_owner_lat, account_owner_lon, item_owner_latitude, item_owner_longitude)
        print("[Debug/itemdetails] distance: ", distance)

        # >>> look up background color based on the distance
        background_color = get_color_from_distance(cursor, conn, distance)
        print("[Debug/itemdetails] background_color: ", background_color)


    else:  # your own item
        print("[Debug/itemdetails] You are viewing your own item")
        is_my_own_item = True

    return render_template('itemdetails.html',
                           item_no=item_no,
                           title=title,
                           condition=condition,
                           item_owner_first_name=item_owner_first_name,
                           item_owner_last_name=item_owner_last_name,
                           item_onwer_postal_code=item_onwer_postal_code,
                           item_owner_city=item_owner_city,
                           item_owner_state=item_owner_state,
                           distance=distance,
                           background_color=background_color,
                           media=media,
                           video_game_platform=video_game_platform,
                           is_video_game=is_video_game,
                           is_collectible_card_game=is_collectible_card_game,
                           is_playing_card_game=is_playing_card_game,
                           is_board_game=is_board_game,
                           is_computer_game=is_computer_game,
                           number_of_cards=number_of_cards,
                           computer_platform=computer_platform,
                           is_my_own_item=is_my_own_item)


if __name__ == "__main__":
    app.run(debug=True)