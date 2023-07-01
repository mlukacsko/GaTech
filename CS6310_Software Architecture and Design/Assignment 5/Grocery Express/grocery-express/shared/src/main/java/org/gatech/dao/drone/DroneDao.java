package org.gatech.dao.drone;

import org.gatech.dto.Drone;

import java.sql.SQLException;
import java.util.List;

public interface DroneDao {

    List<Drone> getAllAvailableDrones() throws SQLException, ClassNotFoundException;
    List<Drone> getDrones(String storeName) throws SQLException, ClassNotFoundException;

    void makeDrone(Drone drone, String storeName) throws SQLException, ClassNotFoundException;

    List<Drone> getDronesByDroneIdForStore(String droneId, String storeName) throws SQLException, ClassNotFoundException;
    List<Drone> getDroneByOrderId(String storeName, String orderId) throws SQLException, ClassNotFoundException;

    void flyDrone(String storeName, String droneId, String pilotAccountId) throws SQLException, ClassNotFoundException;
}
