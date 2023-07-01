-- MySQL dump 10.13  Distrib 8.0.29, for Win64 (x86_64)
--
-- Host: localhost    Database: tradeplaza
-- ------------------------------------------------------
-- Server version	8.0.29

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `item_playing_card_game`
--

DROP TABLE IF EXISTS `item_playing_card_game`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `item_playing_card_game` (
  `lister_email` varchar(250) NOT NULL,
  `item_no` int NOT NULL,
  PRIMARY KEY (`item_no`,`lister_email`),
  KEY `lister_email` (`lister_email`),
  CONSTRAINT `item_playing_card_game_ibfk_1` FOREIGN KEY (`item_no`) REFERENCES `item` (`item_no`),
  CONSTRAINT `item_playing_card_game_ibfk_2` FOREIGN KEY (`lister_email`) REFERENCES `user` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `item_playing_card_game`
--

LOCK TABLES `item_playing_card_game` WRITE;
/*!40000 ALTER TABLE `item_playing_card_game` DISABLE KEYS */;
INSERT INTO `item_playing_card_game` VALUES ('usr001@gt.edu',2),('usr005@gt.edu',23),('usr008@gt.edu',53),('usr018@gt.edu',133),('usr023@gt.edu',153),('usr023@gt.edu',154),('usr027@gt.edu',183),('usr028@gt.edu',190),('usr030@gt.edu',198),('usr037@gt.edu',253),('usr039@gt.edu',266),('usr049@gt.edu',350),('usr052@gt.edu',368),('usr053@gt.edu',384),('usr055@gt.edu',399),('usr057@gt.edu',422),('usr060@gt.edu',447),('usr063@gt.edu',460),('usr063@gt.edu',463),('usr070@gt.edu',516),('usr073@gt.edu',530),('usr079@gt.edu',571),('usr083@gt.edu',606),('usr086@gt.edu',638),('usr087@gt.edu',642),('usr096@gt.edu',734),('usr097@gt.edu',746),('usr101@gt.edu',765),('usr116@gt.edu',876),('usr118@gt.edu',889),('usr121@gt.edu',912),('usr127@gt.edu',944),('usr129@gt.edu',961),('usr132@gt.edu',983),('usr139@gt.edu',1028),('usr145@gt.edu',1065),('usr146@gt.edu',1078),('usr160@gt.edu',1190),('usr164@gt.edu',1221),('usr165@gt.edu',1230),('usr171@gt.edu',1266),('usr175@gt.edu',1291),('usr178@gt.edu',1310),('usr181@gt.edu',1335),('usr182@gt.edu',1342),('usr182@gt.edu',1347),('usr193@gt.edu',1432),('usr194@gt.edu',1445),('usr204@gt.edu',1528),('usr205@gt.edu',1539),('usr206@gt.edu',1549),('usr218@gt.edu',1647),('usr222@gt.edu',1682),('usr231@gt.edu',1732),('usr248@gt.edu',1857),('usr254@gt.edu',1899),('usr255@gt.edu',1908),('usr257@gt.edu',1919),('usr263@gt.edu',1962),('usr270@gt.edu',2011),('usr276@gt.edu',2067),('usr285@gt.edu',2131),('usr294@gt.edu',2190),('usr298@gt.edu',2218),('usr301@gt.edu',2235),('usr302@gt.edu',2246),('usr311@gt.edu',2298),('usr328@gt.edu',2429),('usr334@gt.edu',2461),('usr335@gt.edu',2472),('usr337@gt.edu',2486),('usr344@gt.edu',2546),('usr348@gt.edu',2568),('usr355@gt.edu',2624),('usr356@gt.edu',2633),('usr360@gt.edu',2664),('usr360@gt.edu',2667),('usr364@gt.edu',2701),('usr369@gt.edu',2733),('usr370@gt.edu',2743),('usr372@gt.edu',2764),('usr373@gt.edu',2775),('usr376@gt.edu',2799),('usr377@gt.edu',2802),('usr384@gt.edu',2858),('usr391@gt.edu',2921),('usr393@gt.edu',2941),('usr399@gt.edu',2992),('usr403@gt.edu',3022),('usr407@gt.edu',3053),('usr408@gt.edu',3062),('usr409@gt.edu',3066),('usr411@gt.edu',3081),('usr411@gt.edu',3084),('usr413@gt.edu',3100),('usr416@gt.edu',3120),('usr418@gt.edu',3132),('usr421@gt.edu',3152),('usr424@gt.edu',3171),('usr436@gt.edu',3239),('usr438@gt.edu',3249),('usr438@gt.edu',3250),('usr443@gt.edu',3285),('usr443@gt.edu',3288),('usr443@gt.edu',3289),('usr452@gt.edu',3354),('usr452@gt.edu',3357),('usr466@gt.edu',3450),('usr470@gt.edu',3471),('usr471@gt.edu',3474),('usr475@gt.edu',3500),('usr477@gt.edu',3522),('usr495@gt.edu',3646),('usr507@gt.edu',3741),('usr510@gt.edu',3766),('usr517@gt.edu',3831),('usr518@gt.edu',3839),('usr526@gt.edu',3899),('usr538@gt.edu',3993),('usr538@gt.edu',3995),('usr555@gt.edu',4108),('usr557@gt.edu',4124),('usr559@gt.edu',4133),('usr563@gt.edu',4160),('usr564@gt.edu',4167),('usr565@gt.edu',4180),('usr585@gt.edu',4327),('usr586@gt.edu',4333),('usr597@gt.edu',4407),('usr599@gt.edu',4419),('usr606@gt.edu',4475),('usr616@gt.edu',4550),('usr618@gt.edu',4575),('usr620@gt.edu',4579),('usr629@gt.edu',4646),('usr631@gt.edu',4665),('usr634@gt.edu',4679),('usr636@gt.edu',4692),('usr661@gt.edu',4903),('usr662@gt.edu',4906),('usr666@gt.edu',4937),('usr667@gt.edu',4945),('usr676@gt.edu',5008),('usr685@gt.edu',5089),('usr686@gt.edu',5093),('usr688@gt.edu',5107),('usr689@gt.edu',5117),('usr690@gt.edu',5121),('usr693@gt.edu',5141),('usr693@gt.edu',5147),('usr703@gt.edu',5199),('usr712@gt.edu',5275),('usr718@gt.edu',5313),('usr723@gt.edu',5354),('usr727@gt.edu',5387),('usr733@gt.edu',5433),('usr736@gt.edu',5454),('usr740@gt.edu',5492),('usr741@gt.edu',5503),('usr757@gt.edu',5633),('usr757@gt.edu',5635),('usr759@gt.edu',5648),('usr759@gt.edu',5651),('usr759@gt.edu',5655),('usr760@gt.edu',5657),('usr760@gt.edu',5660),('usr769@gt.edu',5719),('usr771@gt.edu',5733),('usr777@gt.edu',5784),('usr782@gt.edu',5826),('usr791@gt.edu',5888),('usr795@gt.edu',5916),('usr797@gt.edu',5934),('usr798@gt.edu',5943),('usr811@gt.edu',6041),('usr812@gt.edu',6050),('usr813@gt.edu',6053),('usr817@gt.edu',6083),('usr819@gt.edu',6099),('usr819@gt.edu',6102),('usr821@gt.edu',6108),('usr822@gt.edu',6117),('usr824@gt.edu',6133),('usr826@gt.edu',6145),('usr827@gt.edu',6157),('usr829@gt.edu',6175),('usr832@gt.edu',6195),('usr854@gt.edu',6351),('usr855@gt.edu',6356),('usr860@gt.edu',6387),('usr862@gt.edu',6396),('usr866@gt.edu',6427),('usr867@gt.edu',6433),('usr895@gt.edu',6634),('usr904@gt.edu',6707),('usr911@gt.edu',6761),('usr918@gt.edu',6793),('usr922@gt.edu',6823),('usr923@gt.edu',6830),('usr926@gt.edu',6851),('usr927@gt.edu',6868),('usr936@gt.edu',6933),('usr938@gt.edu',6948),('usr941@gt.edu',6981),('usr946@gt.edu',7019),('usr974@gt.edu',7224),('usr975@gt.edu',7228),('usr975@gt.edu',7229),('usr992@gt.edu',7360),('usr999@gt.edu',7414);
/*!40000 ALTER TABLE `item_playing_card_game` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2022-07-20  0:12:40
