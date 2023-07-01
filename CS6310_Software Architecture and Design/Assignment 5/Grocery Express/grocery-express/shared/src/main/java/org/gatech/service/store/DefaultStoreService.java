package org.gatech.service.store;

import com.google.inject.Inject;
import org.gatech.dao.coupon.CouponDao;
import org.gatech.dao.customer.CustomerDao;
import org.gatech.dao.drone.DroneDao;
import org.gatech.dao.item.ItemDao;
import org.gatech.dao.pilot.DronePilotDao;
import org.gatech.dao.store.StoreDao;
import org.gatech.dao.order.OrderDao;
import org.gatech.dto.*;
import org.gatech.exception.GroceryExpressException;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class DefaultStoreService implements StoreService {

    private final StoreDao storeDao;
    private final ItemDao itemDao;
    private final DroneDao droneDao;
    private final DronePilotDao dronePilotDao;
    private final OrderDao orderDao;
    private final CustomerDao customerDao;
    private final CouponDao couponDao;

    @Inject
    public DefaultStoreService(
            StoreDao storeDao,
            ItemDao itemDao,
            DroneDao droneDao,
            DronePilotDao dronePilotDao,
            OrderDao orderDao,
            CustomerDao customerDao,
            CouponDao couponDao
    ) {
        this.storeDao = storeDao;
        this.itemDao = itemDao;
        this.droneDao = droneDao;
        this.dronePilotDao = dronePilotDao;
        this.orderDao = orderDao;
        this.customerDao = customerDao;
        this.couponDao = couponDao;
    }


    @Override
    public List<Store> getStores() throws SQLException, ClassNotFoundException {
        return storeDao.getStores();
    }

    /**
     * Calls dao layer to create store in the database
     *
     * @param store - instance to be added to db
     * @throws SQLException           - is thrown if SQL is invalid
     * @throws ClassNotFoundException
     */
    @Override
    public void createStore(Store store) throws GroceryExpressException, SQLException, ClassNotFoundException {
        if (storeExists(store.getName())) {
            throw new GroceryExpressException("ERROR:store_identifier_already_exists");
        }
        storeDao.createStore(store);
    }

    public void sellItem(Item item, String storeName) throws SQLException, ClassNotFoundException, GroceryExpressException {
        if (!storeExists(item.getName())) {
            throw new GroceryExpressException("ERROR:store_identifier_does_not_exist");
        }
        if (itemExists(item.getItemName(),storeName)){
            throw new GroceryExpressException("ERROR:item_identifier_already_exists");
        }
        itemDao.sellItem(item);
    }

    @Override
    public List<Item> getItems(String storeName) throws SQLException, ClassNotFoundException, GroceryExpressException {
        if (!storeExists(storeName)) {
            throw new GroceryExpressException("ERROR:store_identifier_does_not_exist");
        }
        return itemDao.getItems(storeName);
    }

    @Override
    public void makeDrone(String storeName, Drone drone) throws SQLException, ClassNotFoundException, GroceryExpressException {
        if (!storeExists(storeName)) {
            throw new GroceryExpressException("ERROR:store_identifier_does_not_exist");
        }
        if (droneExists(Integer.parseInt(drone.getDroneID()),storeName)) {
            throw new GroceryExpressException("ERROR:drone_identifier_already_exists");
        }
        droneDao.makeDrone(drone,storeName);

    }

    public List<Drone> getDrones(String storeName) throws SQLException, ClassNotFoundException, GroceryExpressException {
        if (!storeExists(storeName)) {
            throw new GroceryExpressException("ERROR:store_identifier_does_not_exist");
        }
        return droneDao.getDrones(storeName);
    }

    /**
     * @param storeName
     * @param droneId
     * @param pilotId
     * @throws SQLException
     * @throws ClassNotFoundException
     */
    @Override
    public void flyDrone(String storeName, String droneId, String pilotId) throws SQLException, ClassNotFoundException, GroceryExpressException {
        if (!storeExists(storeName)) {
            throw new GroceryExpressException("ERROR:store_identifier_does_not_exist");
        }

        if (!droneExists(Integer.parseInt(droneId), storeName)) {
            throw new GroceryExpressException("ERROR:drone_identifier_does_not_exist");
        }

        if (!pilotExists(pilotId)) {
            throw new GroceryExpressException("ERROR:pilot_identifier_does_not_exist");
        }

        droneDao.flyDrone(storeName, droneId, pilotId);
    }

    public void makeOrder(String storeName, Order order) throws SQLException, ClassNotFoundException, GroceryExpressException {
        if (!storeExists(storeName)) {
            throw new GroceryExpressException("ERROR:store_identifier_does_not_exist");
        }
        if (orderExists(storeName, order.getOrderID())) {
            throw new GroceryExpressException("ERROR:order_identifier_already_exists");
        }
        if (!droneExists(order.getDroneID(), storeName)) {
            throw new GroceryExpressException("ERROR:drone_identifier_does_not_exist");
        }
        if (!customerExists(order.getCustomerID())) {
            throw new GroceryExpressException("ERROR:customer_identifier_does_not_exist");
        }

        // If a drone is not available because it is delivering another order,
        // time must pass before the drone is available to be associated with
        // a new order.
        if (getIsDroneServicingOrderByDroneId(storeName, order.getDroneID())) {
            throw new GroceryExpressException("ERROR:drone_servicing_order");
        }

        orderDao.makeOrder(storeName,order);
    }

    public void transferOrder(String storeName, String orderId, String droneID) throws SQLException, ClassNotFoundException, GroceryExpressException {
        if (!storeExists(storeName)) {
            throw new GroceryExpressException("ERROR:store_identifier_does_not_exist");
        }

        if (!orderExists(storeName, orderId)) {
            throw new GroceryExpressException("ERROR:order_identifier_does_not_exist");
        }

        int droneId = Integer.parseInt(droneID);

        if (!droneExists(droneId, storeName)) {
            throw new GroceryExpressException("ERROR:drone_identifier_does_not_exist");
        }

        if (orderIsAssignedToDrone(storeName, orderId, droneID)) {
            throw new GroceryExpressException("OK:new_drone_is_current_drone_no_change");
        }

        if (!droneHasCapacityForOrder(storeName, droneID, orderId)) {
            throw new GroceryExpressException("ERROR:new_drone_does_not_have_enough_capacity");
        }

        orderDao.transferOrder(storeName, orderId, droneID);
    }

    public void purchaseOrder(String storeName, String orderID, Optional<Boolean> applyCoupon) throws SQLException, ClassNotFoundException, GroceryExpressException {
        if (!storeExists(storeName)) {
            throw new GroceryExpressException("ERROR:store_identifier_does_not_exist");
        }
        if (!orderExists(storeName, orderID)) {
            throw new GroceryExpressException("ERROR:order_identifier_does_not_exist");
        }

        if (droneNeedsPilot(storeName, orderID)) {
            throw new GroceryExpressException("ERROR:drone_needs_pilot");
        }

        if (droneNeedsFuel(storeName, orderID)) {
            throw new GroceryExpressException("ERROR:drone_needs_fuel");
        }

        // If a drone is not available because it is delivering another order,
        // time must pass before the drone is available to allow the purchase
        // to complete.
        if (getIsDroneServicingOrderByOrderId(storeName, orderID)) {
            throw new GroceryExpressException("ERROR:drone_servicing_order");
        }

        List<Coupon> coupons = new ArrayList<>();

        if (applyCoupon.isPresent() && applyCoupon.get()) {
            coupons = couponDao.getApplicableCoupons(storeName, orderID);
        }

        orderDao.purchaseOrder(storeName, orderID, coupons);
    }

    public void cancelOrder(String storeName, String orderID) throws SQLException, ClassNotFoundException, GroceryExpressException {
        if (!storeExists(storeName)) {
            throw new GroceryExpressException("ERROR:store_identifier_does_not_exist");
        }
        if (!orderExists(storeName, orderID)) {
            throw new GroceryExpressException("ERROR:order_identifier_does_not_exist");
        }
        orderDao.cancelOrder(storeName,orderID);
    }

    public List<Order> getOrder(String storeName) throws SQLException, ClassNotFoundException, GroceryExpressException {
        if (!storeExists(storeName)) {
            throw new GroceryExpressException("ERROR:store_identifier_does_not_exist");
        }
        List<Order> orders = orderDao.getOrders(storeName);

        for (Order order : orders) {
            List<OrderItem> items = itemDao.getItemsByOrder(storeName, order.getOrderID());
            order.setItems(items);
        }
        return orders;
    }

    /**
     * @param storeName
     * @param orderIdentifier
     * @param itemName
     * @param quantity
     * @param unitPrice
     * @throws SQLException
     * @throws ClassNotFoundException
     */
    @Override
    public void requestItem(
            String storeName,
            String orderIdentifier,
            String itemName,
            int quantity,
            int unitPrice
    ) throws SQLException, ClassNotFoundException, GroceryExpressException {

        // check that store exists
        if (!storeExists(storeName)) {
            throw new GroceryExpressException("ERROR:store_identifier_does_not_exist");
        }

        // check that order exists
        if (!orderExists(storeName, orderIdentifier)) {
            throw new GroceryExpressException("ERROR:order_identifier_does_not_exist");
        }

        // check that item exists
        if (!itemExists(itemName, storeName)) {
            throw new GroceryExpressException("ERROR:item_identifier_does_not_exist");
        }

        // check that item has not already been added to the order
        if (itemAlreadyOrdered(storeName, orderIdentifier, itemName)){
            throw new GroceryExpressException("ERROR:item_already_ordered");
        }

        // check that the customer can afford the new item
        if (!canCustomerAffordNewItem(storeName, orderIdentifier, quantity, unitPrice)) {
            throw new GroceryExpressException("ERROR:customer_cant_afford_new_item");
        }

        if (!canDroneCarryNewItem(storeName, orderIdentifier, itemName, quantity)) {
            throw new GroceryExpressException("ERROR:drone_cant_carry_new_item");
        }

        itemDao.addItemToOrder(storeName, orderIdentifier, itemName, quantity, unitPrice);
    }


    private boolean storeExists(String storeName) throws SQLException, ClassNotFoundException {
        List<Store> stores = storeDao.getStoresByName(storeName);
        return !stores.isEmpty();
    }

    private boolean itemExists(String itemName, String storeName) throws SQLException, ClassNotFoundException{
        List<Item> item = itemDao.getItemByName(itemName, storeName);
        return (!item.isEmpty());
    }

    private boolean droneExists(int droneId, String storeName) throws SQLException, ClassNotFoundException {
        List<Drone> drones = droneDao.getDronesByDroneIdForStore(String.valueOf(droneId), storeName);
        return !drones.isEmpty();
    }

    private boolean pilotExists(String pilotId) throws SQLException, ClassNotFoundException {
        List<DronePilot> dronePilots = dronePilotDao.getDronePilotsByAccountIdentifier(pilotId);
        return !dronePilots.isEmpty();
    }
    private boolean customerExists(String accountID) throws SQLException, ClassNotFoundException {
        List<Customer> customer = customerDao.getCustomersByAccountId(accountID);
        return !customer.isEmpty();
    }

    private boolean orderExists(String storeName, String orderID) throws SQLException, ClassNotFoundException {
        List<Order> order = orderDao.getOrderByOrderID(storeName, orderID);
        return !order.isEmpty();
    }
    private boolean droneNeedsFuel(String storeName, String orderID) throws SQLException, ClassNotFoundException {
        int netEnergyCost = orderDao.getDroneNetEnergyCostToDeliverOrder(storeName, orderID);
        List<Drone> drone = droneDao.getDroneByOrderId(storeName, orderID);
        return drone.get(0).getNumDeliveriesBeforeMaintenance() < netEnergyCost;
    }

    private boolean getIsDroneServicingOrderByOrderId(String storeName, String orderID) throws SQLException, ClassNotFoundException {
        List<Drone> drone = droneDao.getDroneByOrderId(storeName, orderID);
        return !drone.get(0).getWaitTime().equals("00:00:00");
    }

    private boolean getIsDroneServicingOrderByDroneId(String storeName, int droneId) throws SQLException, ClassNotFoundException {
        List<Drone> drone = droneDao.getDronesByDroneIdForStore(String.valueOf(droneId), storeName);
        return !drone.get(0).getWaitTime().equals("00:00:00");
    }

    private boolean droneNeedsPilot(String storeName, String orderID) throws SQLException, ClassNotFoundException {
        List<Order> order = orderDao.getOrderWithPilotAssigned(storeName, orderID);
        return order.isEmpty();
    }

    private boolean itemAlreadyOrdered(String storeName, String orderIdentifier, String itemName) throws SQLException, ClassNotFoundException {
        List<Item> items = itemDao.getItemsByOrderIdentifier(storeName, orderIdentifier, itemName);
        return !items.isEmpty();
    }

    private boolean orderIsAssignedToDrone(String storeName, String orderId, String droneId) throws SQLException, ClassNotFoundException{
        List<Order> order = orderDao.getOrderByOrderIDAndDrone(storeName, orderId, droneId);
        return !order.isEmpty();
    }

    private boolean droneHasCapacityForOrder(String storeName, String droneId, String orderId) throws SQLException, ClassNotFoundException {
        List<Drone> drones = droneDao.getDronesByDroneIdForStore(droneId, storeName);
        int weightCapacity = drones.get(0).getWeightCapacity();

        int newOrderWeight = orderDao.getOrderWeight(storeName, orderId);
        int currentOrderWeight = orderDao.getCurrentOrderWeight(storeName, droneId);

        return weightCapacity - newOrderWeight - currentOrderWeight >= 0;
    }
    private boolean canDroneCarryNewItem(String storeName, String orderIdentifier, String itemName, int quantity) throws SQLException, ClassNotFoundException {
        List<Drone> drone = droneDao.getDroneByOrderId(storeName, orderIdentifier);
        List<Item> item = itemDao.getItemByName(itemName, storeName);
        int remainingCapacity = drone.get(0).getRemainingCapacity();
        int additionalWeight = (quantity * item.get(0).getItemWeight());
        return remainingCapacity - additionalWeight >= 0;
    }

    private boolean canCustomerAffordNewItem(String storeName, String orderIdentifier, int quantity, int unitPrice) throws SQLException, ClassNotFoundException {
        int futureOrderCost = (quantity * unitPrice);
        return customerDao.canCustomerAffordNewItem(storeName, orderIdentifier, futureOrderCost);
    }
}
