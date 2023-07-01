package org.gatech.service.store;

import org.gatech.dto.Drone;
import org.gatech.dto.Item;
import org.gatech.dto.Store;
import org.gatech.dto.Order;
import org.gatech.exception.GroceryExpressException;

import java.sql.SQLException;
import java.util.List;
import java.util.Optional;

public interface StoreService {

    /**
     * returns a list of stores from the dao layer
     * @return
     * @throws SQLException
     * @throws ClassNotFoundException
     */
    List<Store> getStores() throws SQLException, ClassNotFoundException;

    /**
     * Calls dao layer to create store in the database
     * @param store - instance to be added to db
     * @throws SQLException - is thrown if SQL is invalid
     * @throws ClassNotFoundException
     */
    void createStore(Store store) throws SQLException, ClassNotFoundException, GroceryExpressException;

    void sellItem(Item item, String storeName) throws SQLException, ClassNotFoundException, GroceryExpressException;

    List<Item> getItems(String storeName) throws SQLException, ClassNotFoundException, GroceryExpressException;

    void makeDrone(String storeName, Drone drone) throws  SQLException, ClassNotFoundException, GroceryExpressException;

    List<Drone> getDrones(String storeName) throws SQLException, ClassNotFoundException, GroceryExpressException;

    void flyDrone(String storeName, String droneId, String pilotId) throws SQLException, ClassNotFoundException, GroceryExpressException;

    void makeOrder(String storeName, Order order) throws  SQLException, ClassNotFoundException, GroceryExpressException;

    List<Order> getOrder(String storeName) throws SQLException, ClassNotFoundException, GroceryExpressException;
    void transferOrder(String storeName, String orderID, String droneId) throws SQLException, ClassNotFoundException, GroceryExpressException;
    void cancelOrder(String storeName, String orderID) throws SQLException, ClassNotFoundException, GroceryExpressException;
    void purchaseOrder(String storeName, String orderID, Optional<Boolean> applyCoupon) throws  SQLException, ClassNotFoundException, GroceryExpressException;
    void requestItem(String storeName, String orderIdentifier, String itemName, int quantity, int unitPrice) throws SQLException, ClassNotFoundException, GroceryExpressException;
}
