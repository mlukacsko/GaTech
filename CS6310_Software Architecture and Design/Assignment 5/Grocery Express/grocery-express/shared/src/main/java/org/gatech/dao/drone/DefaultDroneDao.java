package org.gatech.dao.drone;

import com.google.inject.Inject;
import org.gatech.dbconnect.ConnectionManager;
import org.gatech.dto.Drone;

import java.sql.SQLException;
import java.util.List;

public class DefaultDroneDao implements DroneDao {

    private final ConnectionManager<Drone> manager;
    private final DroneRowMapper droneRowMapper;

    @Inject
    public DefaultDroneDao(ConnectionManager<Drone> manager, DroneRowMapper droneRowMapper) {
        this.manager = manager;
        this.droneRowMapper = droneRowMapper;
    }

    public List<Drone> getAllAvailableDrones() throws SQLException, ClassNotFoundException {
        String selectTotalOrderWeight = "(SELECT SUM(ge_line_item.quantity * ge_item.weight) " +
                "FROM ge_order " +
                "INNER JOIN ge_line_item ON ge_line_item.order_id = ge_order.id " +
                "INNER JOIN ge_item ON ge_line_item.item_id = ge_item.id " +
                "INNER JOIN ge_drone ON ge_order.drone_id = ge_drone.id " +
                "WHERE ge_order.drone_id = ge_drone.id " +
                "AND ge_drone.wait_time = '00:00:00'" +
        ")";

        String selectOrderCount = "(SELECT COUNT(*) FROM ge_order WHERE ge_order.drone_id = ge_drone.id)";

        return manager.getEntities("SELECT " +
                "drone_id, weight_capacity, remaining_delivery_count, " +
                selectOrderCount + " AS assigned_orders_count, " +
                selectTotalOrderWeight + " AS assigned_orders_weight, " +
                "first_name, last_name, wait_time " +
                "FROM ge_drone " +
                "INNER JOIN ge_store ON ge_drone.store_id = ge_store.id " +
                "LEFT OUTER JOIN ge_pilot_person ON ge_drone.pilot_id = ge_pilot_person.pilot_id " +
                "ORDER BY drone_id ASC;", droneRowMapper);
    }

    @Override
    public List<Drone> getDrones(String storeName) throws SQLException, ClassNotFoundException {
        String selectTotalOrderWeight = "(SELECT SUM(ge_line_item.quantity * ge_item.weight) " +
                "FROM ge_order " +
                "INNER JOIN ge_line_item ON ge_line_item.order_id = ge_order.id " +
                "INNER JOIN ge_item ON ge_line_item.item_id = ge_item.id " +
                "WHERE ge_order.drone_id = ge_drone.id" +
                ")";

        String selectOrderCount = "(SELECT COUNT(*) FROM ge_order WHERE ge_order.drone_id = ge_drone.id)";
        return manager.getEntities("SELECT " +
                "drone_id, weight_capacity, remaining_delivery_count, " +
                selectOrderCount + " AS assigned_orders_count, " +
                selectTotalOrderWeight + " AS assigned_orders_weight, " +
                "first_name, last_name, wait_time " +
                "FROM ge_drone " +
                "INNER JOIN ge_store ON ge_drone.store_id = ge_store.id " +
                "LEFT OUTER JOIN ge_pilot_person ON ge_drone.pilot_id = ge_pilot_person.pilot_id " +
                "WHERE ge_store.name = '" + storeName + "' " +
                "ORDER BY drone_id ASC;", droneRowMapper);
    }


    public void makeDrone(Drone drone, String storeName) throws SQLException, ClassNotFoundException {
        manager.executeSql(
                "INSERT INTO ge_drone(store_id,drone_id,weight_capacity,remaining_delivery_count) VALUES("
                        + "(SELECT id FROM ge_store WHERE name = '" + storeName + "'),"
                        + "'" + drone.getDroneID() + "'" + ","
                        + drone.getWeightCapacity() + ","
                        + drone.getNumDeliveriesBeforeMaintenance()
                        + ");"
        );
    }

    /**
     * @param droneId
     * @param storeName
     * @return
     * @throws SQLException
     * @throws ClassNotFoundException
     */
    @Override
    public List<Drone> getDronesByDroneIdForStore(String droneId, String storeName) throws SQLException, ClassNotFoundException {
        String selectOrderCount = "(SELECT COUNT(*) FROM ge_order WHERE ge_order.drone_id = ge_drone.id)";
        String selectTotalOrderWeight = "(SELECT SUM(ge_line_item.quantity * ge_item.weight) " +
                "FROM ge_order " +
                "INNER JOIN ge_line_item ON ge_line_item.order_id = ge_order.id " +
                "INNER JOIN ge_item ON ge_line_item.item_id = ge_item.id " +
                "WHERE ge_order.drone_id = ge_drone.id" +
                ")";
        return manager.getEntities("SELECT " +
                "drone_id, weight_capacity, remaining_delivery_count, " +
                selectOrderCount + " AS assigned_orders_count, " +
                selectTotalOrderWeight + " AS assigned_orders_weight, " +
                "first_name, last_name, wait_time " +
                "FROM ge_drone " +
                "INNER JOIN ge_store on ge_drone.store_id = ge_store.id " +
                "LEFT OUTER JOIN ge_pilot_person on ge_drone.pilot_id = ge_pilot_person.pilot_id " +
                "WHERE drone_id='" + droneId + "' " +
                "AND ge_store.name = '" + storeName + "';", droneRowMapper);
    }

    @Override
    public List<Drone> getDroneByOrderId(String storeName, String orderId) throws SQLException, ClassNotFoundException {
        String selectOrderCount = "(SELECT COUNT(*) FROM ge_order WHERE ge_order.drone_id = ge_drone.id)";
        String selectTotalOrderWeight = "(SELECT SUM(ge_line_item.quantity * ge_item.weight) " +
                "FROM ge_order " +
                "INNER JOIN ge_line_item ON ge_line_item.order_id = ge_order.id " +
                "INNER JOIN ge_item ON ge_line_item.item_id = ge_item.id " +
                "WHERE ge_order.drone_id = ge_drone.id" +
                ")";
        return manager.getEntities("SELECT " +
                        "ge_drone.drone_id, weight_capacity, remaining_delivery_count, " +
                        selectOrderCount + " AS assigned_orders_count, " +
                        selectTotalOrderWeight + " AS assigned_orders_weight, " +
                        "first_name, last_name, wait_time " +
                        "FROM ge_drone " +
                        "INNER JOIN ge_store on ge_drone.store_id = ge_store.id " +
                        "INNER JOIN ge_order on ge_order.drone_id = ge_drone.id " +
                        "LEFT OUTER JOIN ge_pilot_person on ge_drone.pilot_id = ge_pilot_person.pilot_id " +
                        "WHERE ge_order.order_id='" + orderId + "' " +
                        "AND ge_store.name = '" + storeName + "';",
                droneRowMapper);
    }

    /**
     * @param storeName
     * @param droneId
     * @param pilotAccountId
     * @throws SQLException
     * @throws ClassNotFoundException
     */
    @Override
    public void flyDrone(String storeName, String droneId, String pilotAccountId) throws SQLException, ClassNotFoundException {

        // set the pilot_id of the prev drone to null since the pilot can only fly one drone
        manager.executeSql(
                "UPDATE ge_drone " +
                        "SET pilot_id = NULL " +
                        "FROM " +
                        "(SELECT * FROM ge_pilot WHERE account_id = '" + pilotAccountId + "') AS drone_pilot_query " +
                        "WHERE pilot_id=drone_pilot_query.id;"
        );

        // assign the pilot to the drone
        manager.executeSql(
                "UPDATE ge_drone " +
                        "SET pilot_id=drone_pilot_query.id " +
                        "FROM " +
                        "(SELECT * FROM ge_store WHERE name='" + storeName + "') AS store_query, " +
                        "(SELECT * FROM ge_pilot WHERE account_id = '" + pilotAccountId + "') AS drone_pilot_query " +
                        "WHERE drone_id = '" + droneId + "'" +
                        "  AND store_id = store_query.id;"
        );
    }
}
