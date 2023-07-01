package org.gatech.dao.order;

import org.gatech.dto.Coupon;
import org.gatech.dto.Order;

import java.sql.SQLException;
import java.util.List;
import java.util.Optional;

public interface OrderDao {

    List<Order> getOrders(String storeName) throws SQLException, ClassNotFoundException;

    void makeOrder(String storeName, Order order) throws SQLException, ClassNotFoundException;
    void transferOrder(String storeName, String orderId, String droneId) throws SQLException, ClassNotFoundException;
    void cancelOrder(String storeName, String orderID) throws SQLException, ClassNotFoundException;
    int getOrderWeight(String storeName, String orderID) throws SQLException, ClassNotFoundException;
    int getCurrentOrderWeight(String storeName, String droneId) throws SQLException, ClassNotFoundException;
    List<Order> getOrderByOrderID(String storeName, String orderID) throws SQLException, ClassNotFoundException;
    List<Order> getOrderByOrderIDAndDrone(String storeName, String orderId, String droneId) throws SQLException, ClassNotFoundException;
    void purchaseOrder(String storeName, String orderID, List<Coupon> coupons) throws SQLException, ClassNotFoundException;
    int getDroneNetEnergyCostToDeliverOrder(String storeName, String orderId) throws SQLException, ClassNotFoundException;
    List<Order> getOrderWithPilotAssigned(String storeName, String orderID) throws SQLException, ClassNotFoundException;
    int getOrderTotalCost(String storeName, String orderIdentifier) throws SQLException, ClassNotFoundException;
}
