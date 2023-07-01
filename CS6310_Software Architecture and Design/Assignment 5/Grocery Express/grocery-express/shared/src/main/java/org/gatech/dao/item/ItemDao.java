package org.gatech.dao.item;

import org.gatech.dto.*;

import java.sql.SQLException;
import java.util.List;

public interface ItemDao {

    List<Item> getItems(String storeName) throws SQLException, ClassNotFoundException;

    void sellItem(Item item) throws SQLException, ClassNotFoundException;

    List<Item> getItemByName(String itemName, String storeName) throws SQLException, ClassNotFoundException;

    List<Item> getItemsByOrderIdentifier(String storeName, String orderIdentifier, String itemName) throws SQLException, ClassNotFoundException;

    void addItemToOrder(String storeName, String orderIdentifier, String itemName, int quantity, int unitPrice) throws ClassNotFoundException, SQLException;

    List<Item> itemsExist(String storeName, String orderID) throws SQLException, ClassNotFoundException;

    List<OrderItem> getItemsByOrder(String storeName, String orderId) throws SQLException, ClassNotFoundException;

}
