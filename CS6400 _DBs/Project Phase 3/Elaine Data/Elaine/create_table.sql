CREATE TABLE tradeplaza.Location_Lookup (
  postal_code varchar(250) NOT NULL,
  city varchar(250) NOT NULL,
  state varchar(250) NOT NULL,
  latitude float(8) NOT NULL,
  longitude float(8) NOT NULL,
  PRIMARY KEY (postal_code)
);


CREATE TABLE tradeplaza.User (
  email varchar(250) NOT NULL,
  password varchar(250) NOT NULL,
  first_name varchar(250) NOT NULL,
  last_name varchar(250) NOT NULL,
  nickname varchar(250) NOT NULL,
  postal_code varchar(250) NOT NULL,
  PRIMARY KEY (email),
  UNIQUE(nickname),
  FOREIGN KEY (postal_code) REFERENCES Location_Lookup (postal_code)
);

INSERT INTO Location_Lookup VALUES ('55302','Annandale','MN',45.246631,-94.11692);
INSERT INTO Location_Lookup VALUES ('20227','Washington','DC',38.893311,-77.014647);
INSERT INTO Location_Lookup VALUES ('14043','Depew','NY',42.904958,-78.7006);

CREATE TABLE tradeplaza.Platform (
  name varchar(250) NOT NULL,
  friendly_platform_name varh(250) NOT NULL,
  PRIMARY KEY (name))

INSERT INTO tradeplaza.Platform VALUES ('nintendo','Nintendo');
INSERT INTO tradeplaza.Platform VALUES ('xbox','Xbox');
INSERT INTO tradeplaza.Platform VALUES ('playstation','PlayStation');

CREATE TABLE tradeplaza.`Item` (
  `lister_email` varchar(250) NOT NULL,
  `title` varchar(250) NOT NULL,
  `item_no` int NOT NULL,
  `condition` varchar(250) NOT NULL,
  `description` varchar(250) DEFAULT NULL,
  PRIMARY KEY (`item_no`,`lister_email`),
  KEY `lister_email` (`lister_email`),
  CONSTRAINT `item_ibfk_1` FOREIGN KEY (`lister_email`) REFERENCES `User` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci

CREATE TABLE tradeplaza.Item_Collectable_Card_Game (
  lister_email varchar(250) NOT NULL,
  item_no int(16) NOT NULL,
  number_of_cards int(16) NOT NULL,
  PRIMARY KEY (item_no, lister_email),
  FOREIGN KEY (item_no) REFERENCES Item (item_no),
  FOREIGN KEY (lister_email) REFERENCES `User` (email)
);

CREATE TABLE tradeplaza.Item_Board_Game (
  lister_email varchar(250) NOT NULL,
  item_no int(16) NOT NULL,
  PRIMARY KEY (item_no, lister_email),
  FOREIGN KEY (item_no) REFERENCES Item (item_no),
  FOREIGN KEY (lister_email) REFERENCES `User` (email)
);

CREATE TABLE tradeplaza.Item_Playing_Card_Game (
  lister_email varchar(250) NOT NULL,
  item_no int(16) NOT NULL,
  PRIMARY KEY (item_no, lister_email),
  FOREIGN KEY (item_no) REFERENCES Item (item_no),
  FOREIGN KEY (lister_email) REFERENCES `User` (email)
);

CREATE TABLE tradeplaza.Item_Video_Game (
  lister_email varchar(250) NOT NULL,
  item_no int(16) NOT NULL,
  platform varchar(250) NOT NULL,
  media varchar(250) NOT NULL,
  PRIMARY KEY (item_no, lister_email, platform),
  FOREIGN KEY (item_no) REFERENCES Item (item_no),
  FOREIGN KEY (lister_email) REFERENCES `User` (email),
  FOREIGN KEY (platform) REFERENCES Platform (`name`)
);

CREATE TABLE tradeplaza.Item_Computer_Game (
  lister_email varchar(250) NOT NULL,
  item_no int(16) NOT NULL,
  platform varchar(250) NOT NULL,
  PRIMARY KEY (item_no, lister_email),
  FOREIGN KEY (item_no) REFERENCES Item (item_no),
  FOREIGN KEY (lister_email) REFERENCES `User` (email)
);

CREATE TABLE Trade (
  proposer_item_no int(16) NOT NULL,
  counterparty_item_no int(16) NOT NULL,
  proposer_email varchar(250) NOT NULL,
  counterparty_email varchar(250) NOT NULL,
  proposed_date datetime NOT NULL,
  accept_reject_date datetime NULL,
  status varchar(250) NOT NULL,
  auto_trade_id varchar(250),
  PRIMARY KEY (auto_trade_id),
  FOREIGN KEY (proposer_item_no) REFERENCES Item (item_no),
  FOREIGN KEY (counterparty_item_no) REFERENCES Item (item_no),
  FOREIGN KEY (proposer_email) REFERENCES `User` (email),
  FOREIGN KEY (counterparty_email) REFERENCES `User` (email)
);
