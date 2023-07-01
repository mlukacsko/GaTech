package org.gatech.dao.order;

import com.google.inject.Inject;
import org.gatech.dao.location.LocationDao;
import org.gatech.dao.settings.SettingsDao;
import org.gatech.dbconnect.ConnectionManager;
import org.gatech.dto.Coupon;
import org.gatech.dto.Drone;
import org.gatech.dto.Item;
import org.gatech.dto.Location;
import org.gatech.dto.Order;
import org.gatech.dto.Settings;

import java.sql.SQLException;
import java.util.Date;
import java.util.List;

public class DefaultOrderDao implements OrderDao {

    private final ConnectionManager<Order> manager;
    private final OrderRowMapper orderRowMapper;
    private final LocationDao locationDao;
    private final SettingsDao settingsDao;
    private final int DELIVERY_FEE_PER_HOUR = 1;

    @Inject
    public DefaultOrderDao(ConnectionManager<Order> manager, OrderRowMapper orderRowMapper, LocationDao locationDao, SettingsDao settingsDao) {
        this.manager = manager;
        this.orderRowMapper = orderRowMapper;
        this.locationDao = locationDao;
        this.settingsDao = settingsDao;
    }

    @Override
    public List<Order> getOrders(String storeName) throws SQLException, ClassNotFoundException {
        return manager.getEntities("SELECT * FROM ge_order o, ge_store s " +
                "WHERE s.name = '" + storeName + "' " +
                "AND o.store_id = s.id " +
                "ORDER BY o.order_id ASC;", orderRowMapper);
    }

    public void makeOrder(String storeName, Order order) throws SQLException, ClassNotFoundException {
        String selectStoreId = "(SELECT id FROM ge_store WHERE name = '" + storeName + "')";
        String selectDroneId = "(SELECT ge_drone.id as id FROM ge_drone INNER JOIN ge_store on ge_drone.store_id = ge_store.id WHERE drone_id = '" + order.getDroneID() + "' AND ge_store.name = '" + storeName + "')";
        String selectCustomerId = "(SELECT id FROM ge_customer WHERE account_id = '" + order.getCustomerID() + "')";
        manager.executeSql(
                "INSERT INTO ge_order(store_id,order_id,drone_id,customer_id) VALUES(" +
                        selectStoreId + "," +
                        "'" + order.getOrderID() + "'" + "," +
                        selectDroneId + "," +
                        selectCustomerId + ");"
        );
    }

    public void transferOrder(String storeName, String orderId, String droneId) throws SQLException, ClassNotFoundException {
        String selectStoreId = "(SELECT id FROM ge_store WHERE name = '" + storeName + "')";
        String selectDroneId = "(SELECT ge_drone.id FROM ge_drone " +
                "INNER JOIN ge_store on ge_drone.store_id = ge_store.id " +
                "WHERE drone_id = '" + droneId + "' " +
                "AND ge_store.name = '" + storeName + "' " +
        ")";

        manager.executeSql(
                "UPDATE ge_order set drone_id = " + selectDroneId + " " +
                        "WHERE ge_order.store_id = " + selectStoreId + " " +
                        "AND ge_order.order_id = '" + orderId + "'"
        );

        // Update the number of successful transfers for this store
        manager.executeSql(
                "UPDATE ge_store SET " +
                        "transferred_order_count = (transferred_order_count + 1) " +
                        "WHERE name = '" + storeName + "';"
        );
    }

    private void deleteOrder(String storeName, String orderID) throws SQLException, ClassNotFoundException {
        manager.executeSql(
                "DELETE FROM ge_line_item " +
                        "USING ge_order " +
                        "WHERE ge_order.id = ge_line_item.order_id " +
                        "AND ge_order.order_id = '" + orderID + "' " +
                        "AND ge_order.store_id = (SELECT id FROM ge_store WHERE name = '" + storeName + "');"
        );

        manager.executeSql(
                "DELETE FROM ge_order WHERE order_id = '" + orderID + "' " +
                        "AND store_id = (SELECT id FROM ge_store WHERE name = '" + storeName + "');"
        );
    }

    public void cancelOrder(String storeName, String orderID) throws SQLException, ClassNotFoundException {
        deleteOrder(storeName, orderID);
    }

    private void updateDroneForOrderDelivery(String storeName, String orderID, int waitTimeHours, int amount) throws SQLException, ClassNotFoundException {
        manager.executeSql(
                "UPDATE ge_drone SET " +
                        "remaining_delivery_count = (remaining_delivery_count - " + amount + "), " +
                        "wait_time = '" + waitTimeHours + " hours' " +
                        "WHERE id = (SELECT ge_order.drone_id FROM ge_order " +
                        "INNER JOIN ge_drone ON ge_order.drone_id = ge_drone.id " +
                        "INNER JOIN ge_store ON ge_order.store_id = ge_store.id " +
                        "WHERE ge_store.name = '" + storeName + "' " +
                        "AND ge_order.order_id = '" + orderID + "' " +
                        ");"
        );
    }

    private int getDroneTimeToDeliverHours(String storeName, String orderId) throws SQLException, ClassNotFoundException{
        // First, check if both the customer and the store have a location associated with them. If either one
        // does not have a location, we will perform the delivery in "curbside pickup" mode; where the drone
        // will take 0 hours, effectively, to deliver the order.
        List<Location> storeLocationResult = locationDao.getStoreLocation(storeName);
        List<Location> customerLocationResult = locationDao.getOrderCustomerLocation(storeName, orderId);
        if (storeLocationResult.isEmpty() || customerLocationResult.isEmpty()) {
            return 0;
        }

        Location storeLocation = storeLocationResult.get(0);
        Location customerLocation = customerLocationResult.get(0);

        // Get the vector length between customer and store
        // For purposes of travel, this length is synonymous with hours of travel for the drone
        // For simplicity we do not do fractional hours, just take the ceiling of the travel time
        // That also serves to give drones some level of emergency fuel available to them.
        double distance = Math.sqrt(
                Math.pow(storeLocation.getX() - customerLocation.getX(), 2) +
                        Math.pow(storeLocation.getY() - customerLocation.getY(), 2)
        );
        int hoursOfTravel = (int) Math.ceil(distance);
        return hoursOfTravel;
    }

    public int getDroneNetEnergyCostToDeliverOrder(String storeName, String orderId) throws SQLException, ClassNotFoundException {
        List<Settings> settings = settingsDao.getSettings();
        int droneEnergyConsumption = settings.get(0).getDroneEnergyConsumption();

        // First, check if both the customer and the store have a location associated with them. If either one
        // does not have a location, we will perform the delivery in "curbside pickup" mode; where the drone
        // will lose exactly one unit of energy (by default, configurable in settings).
        List<Location> storeLocationResult = locationDao.getStoreLocation(storeName);
        List<Location> customerLocationResult = locationDao.getOrderCustomerLocation(storeName, orderId);
        if (storeLocationResult.isEmpty() || customerLocationResult.isEmpty()) {
            return droneEnergyConsumption;
        }

        Location storeLocation = storeLocationResult.get(0);
        Location customerLocation = customerLocationResult.get(0);

        // Get the vector length between customer and store
        // For purposes of travel, this length is synonymous with hours of travel for the drone
        // For simplicity we do not do fractional hours, just take the ceiling of the travel time
        // That also serves to give drones some level of emergency fuel available to them.
        double distance = Math.sqrt(
                Math.pow(storeLocation.getX() - customerLocation.getX(), 2) +
                Math.pow(storeLocation.getY() - customerLocation.getY(), 2)
        );
        int hoursOfTravel = (int) Math.ceil(distance);

        // Get the total cost of energy for the drone to traverse the distance
        int energyCost = droneEnergyConsumption * hoursOfTravel;

        // Get the amount of energy the solar energy drone will gain while traversing that distance
        String clockTime = settings.get(0).getClock();
        int clockHour = Integer.parseInt(clockTime.split(":")[0]);
        double energyOffset = 0;

        for (int i = clockHour; i < clockHour + hoursOfTravel; i++) {
            int hour = i <= 23 ? i : i % 24;
            energyOffset += droneEnergyConsumption * settingsDao.getSolarEnergyMap().get(hour);
        }

        // Compute net energy. We floor the offset of energy gained to avoid
        // over-estimating the energy gained.
        int netEnergyCost = energyCost - (int) Math.floor(energyOffset);
        return netEnergyCost;
    }

    private int doOrderDelivery(String storeName, String orderID) throws SQLException, ClassNotFoundException {
        int waitTimeHours = getDroneTimeToDeliverHours(storeName, orderID);
        int netEnergyCost = getDroneNetEnergyCostToDeliverOrder(storeName, orderID);
        updateDroneForOrderDelivery(storeName, orderID, waitTimeHours, netEnergyCost);
        return waitTimeHours * DELIVERY_FEE_PER_HOUR;
    }

    public void purchaseOrder(String storeName, String orderID, List<Coupon> coupons) throws SQLException, ClassNotFoundException {
        // Increment pilot experience +1
        manager.executeSql(
                "UPDATE ge_pilot SET successful_delivery_count = successful_delivery_count + 1 " +
                "WHERE id = (SELECT pilot_id FROM ge_order " +
                    "INNER JOIN ge_drone ON ge_order.drone_id = ge_drone.id " +
                    "INNER JOIN ge_store ON ge_order.store_id = ge_store.id " +
                    "WHERE ge_store.name = '" + storeName + "' " +
                    "AND ge_order.order_id = '" + orderID + "' " +
                ");"
        );

        // Decrement the drone energy level and update the wait time
        // before the drone is available again.
        int deliveryFee = doOrderDelivery(storeName, orderID);

        // Calculate units in order
        int orderTotal = manager.executeSqlWithReturn(
                "SELECT SUM(li.quantity * li.unit_price) AS id " +
                "FROM ge_line_item li " +
                "INNER JOIN ge_order o ON o.id=li.order_id " +
                "INNER JOIN ge_store s ON s.id=o.store_id " +
                "WHERE s.name='" + storeName + "' " +
                "AND o.order_id='" + orderID + "';"
        ) + deliveryFee;

        // apply discounts from coupons
        for (Coupon coupon : coupons) {
            if (new Date().compareTo(coupon.getExpirationDate()) < 0) { // only apply coupon if it's not expired
                orderTotal = orderTotal - (orderTotal * coupon.getPercentage() / 100);
            }
        }

        // Add cost of order to the store revenue
        // Increment the number of completed purchases by 1
        // Increment the number of overloads for the store
        String droneIdQuery = "(" +
                "SELECT ge_drone.id from ge_order " +
                "INNER JOIN ge_drone ON ge_order.drone_id = ge_drone.id " +
                "INNER JOIN ge_store ON ge_order.store_id = ge_store.id " +
                "WHERE ge_store.name = '" + storeName + "' " +
                "AND ge_order.order_id = '" + orderID + "' " +
                ")";

        int droneOrderCount = manager.executeSqlWithReturn(
                "SELECT count(*) as id FROM ge_order " +
                "INNER JOIN ge_drone on ge_order.drone_id = ge_drone.id " +
                "INNER JOIN ge_store on ge_order.store_id = ge_store.id " +
                "WHERE ge_store.name='" + storeName + "' " +
                "AND ge_order.drone_id = " + droneIdQuery + ";"
        );

        int droneOverloadCount = droneOrderCount - 1;

        manager.executeSql(
                "UPDATE ge_store SET " +
                        "earned_revenue = (earned_revenue + " + orderTotal + "), " +
                        "completed_order_count = (completed_order_count + 1), " +
                        "drone_overload_count = (drone_overload_count + " + droneOverloadCount + ") " +
                        "WHERE name = '" + storeName + "';"
        );

        // Deduct cost of order from customer credit
        manager.executeSql(
                "UPDATE ge_customer SET credits = credits - " + orderTotal + " " +
                "WHERE id = (SELECT customer_id FROM ge_order " +
                        "INNER JOIN ge_store ON ge_order.store_id = ge_store.id " +
                        "WHERE ge_store.name = '" + storeName + "' " +
                        "AND ge_order.order_id = '" + orderID + "' "  +
                ");"
        );

        deleteOrder(storeName, orderID);
    }

    public List<Order> getOrderByOrderID(String storeName, String orderID) throws SQLException, ClassNotFoundException {
        return manager.getEntities("SELECT o.* FROM ge_order o, ge_store s " +
                "WHERE o.store_id = (SELECT id from ge_store WHERE name = '" + storeName + "') " +
                "AND o.order_id = '" + orderID + "' " +
                "AND o.store_id = s.id;", orderRowMapper);
    }

    public List<Order> getOrderWithPilotAssigned(String storeName, String orderID) throws SQLException, ClassNotFoundException {
        return manager.getEntities(
                "SELECT * FROM ge_order " +
                        "INNER JOIN ge_store ON ge_order.store_id = ge_store.id " +
                        "INNER JOIN ge_drone ON ge_order.drone_id = ge_drone.id " +
                        "INNER JOIN ge_pilot ON ge_drone.pilot_id = ge_pilot.id " +
                        "WHERE ge_order.order_id = '" + orderID + "' " +
                        "AND ge_store.name = '" + storeName + "';",
                orderRowMapper
        );
    }

    public int getOrderWeight(String storeName, String orderID) throws SQLException, ClassNotFoundException {
        return manager.executeSqlWithReturn("" +
                "SELECT SUM(li.quantity * i.weight) AS id " +
                "FROM ge_line_item li " +
                "INNER JOIN ge_item  i ON li.item_id = i.id " +
                "INNER JOIN ge_order o ON o.id = li.order_id " +
                "INNER JOIN ge_store s ON s.id = o.store_id " +
                "WHERE s.name='" + storeName + "' " +
                "AND o.order_id='" + orderID + "';");
    }

    public int getCurrentOrderWeight(String storeName, String droneId) throws SQLException, ClassNotFoundException {
        return manager.executeSqlWithReturn("" +
                "SELECT SUM(li.quantity * i.weight) AS id " +
                "FROM ge_line_item li " +
                "INNER JOIN ge_item  i ON li.item_id = i.id " +
                "INNER JOIN ge_order o ON o.id = li.order_id " +
                "INNER JOIN ge_store s ON s.id = o.store_id " +
                "LEFT OUTER JOIN ge_drone d ON d.id = o.drone_id " +
                "WHERE s.name='" + storeName + "' " +
                "AND d.drone_id='" + droneId + "';");
    }

    public List<Order> getOrderByOrderIDAndDrone(String storeName, String orderId, String droneId) throws SQLException, ClassNotFoundException {
        String selectStoreId = "(SELECT id from ge_store WHERE name = '" + storeName + "')";
        String selectDroneId = "(SELECT ge_drone.id FROM ge_drone INNER JOIN ge_store on ge_drone.store_id = ge_store.id WHERE drone_id = '" + droneId + "' AND ge_store.name = '" + storeName + "')";
        return manager.getEntities("SELECT o.* FROM ge_order o, ge_store s " +
                "WHERE o.store_id = " + selectStoreId + " " +
                "AND o.order_id = '" + orderId + "' " +
                "AND o.drone_id = " + selectDroneId + " " +
                "AND o.store_id = s.id;", orderRowMapper);
    }

    /**
     * @param storeName
     * @param orderIdentifier
     * @return
     * @throws SQLException
     * @throws ClassNotFoundException
     */
    @Override
    public int getOrderTotalCost(String storeName, String orderIdentifier) throws SQLException, ClassNotFoundException {
        return manager.executeSqlWithReturn(
                "SELECT SUM(li.quantity * li.unit_price) AS id " +
                        "FROM ge_line_item as li " +
                        "INNER JOIN ge_order o ON o.id=li.order_id " +
                        "INNER JOIN ge_store s ON s.id=o.store_id " +
                        "WHERE s.name='" + storeName + "' " +
                        "  AND o.order_id='" + orderIdentifier + "';"
        );
    }
}
