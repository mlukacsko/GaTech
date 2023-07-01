package org.gatech.dto;

import org.gatech.dbconnect.DatabaseEntity;

public class Drone extends DatabaseEntity {

    private String droneID;
    private int weightCapacity;
    private int numDeliveriesBeforeMaintenance;
    private String pilotFirstName;
    private String pilotLastName;
    private int numberOfOrders;
    private int totalOrderWeight;
    private String waitTime;


    public Drone(String storeName, String droneID, int weightCapacity, int numDeliveriesBeforeMaintenance, int numberOfOrders, int totalOrderWeight, String waitTime) {
        this.droneID = droneID;
        this.weightCapacity = weightCapacity;
        this.numDeliveriesBeforeMaintenance = numDeliveriesBeforeMaintenance;
        this.pilotFirstName = null;
        this.pilotLastName = null;
        this.numberOfOrders = numberOfOrders;
        this.totalOrderWeight = totalOrderWeight;
        this.waitTime = waitTime;
    }

    private Drone(DroneBuilder builder) {
        this.droneID = builder.droneID;
        this.weightCapacity = builder.weightCapacity;
        this.numDeliveriesBeforeMaintenance = builder.numDeliveriesBeforeMaintenance;
        this.pilotFirstName = builder.pilotFirstName;
        this.pilotLastName = builder.pilotLastName;
        this.numberOfOrders = builder.numberOfOrders;
        this.totalOrderWeight = builder.totalOrderWeight;
        this.waitTime = builder.waitTime;
    }


    public String getDroneID() {
        return droneID;
    }
    public void setDroneID(String droneID) {
        this.droneID = droneID;
    }

    public int getWeightCapacity() {
        return weightCapacity;
    }
    public void setWeightCapacity(int weightCapacity ) {
        this.weightCapacity = weightCapacity;
    }

    public int getNumDeliveriesBeforeMaintenance() {return numDeliveriesBeforeMaintenance; }
    public void setNumDeliveriesBeforeMaintenance(int numDeliveriesBeforeMaintenance ) {
        this.numDeliveriesBeforeMaintenance = numDeliveriesBeforeMaintenance;
    }
    public String getPilotName() {
        if (pilotFirstName == null || pilotLastName == null) {
            return "";
        }

        if (pilotFirstName.equals("") || pilotLastName.equals("")) {
            return "";
        }

        return pilotFirstName + "_" + pilotLastName;
    }
    public int getNumberOfOrders() { return numberOfOrders;}
    public void setNumberOfOrders(int numberOfOrders) {
        this.numberOfOrders = numberOfOrders;
    }
    public int getTotalOrderWeight() { return totalOrderWeight;}
    public void setTotalOrderWeight(int totalOrderWeight) { this.totalOrderWeight = totalOrderWeight; }
    public String getWaitTime() { return waitTime; }
    public void setWaitTime(String waitTime) { this.waitTime = waitTime; }
    public int getRemainingCapacity() {
        return weightCapacity - totalOrderWeight;
    }

    @Override
    public String toString() {
        String droneDisplay = "droneID:" + droneID + ",total_cap:" + weightCapacity +
                ",num_orders:" + numberOfOrders + ",remaining_cap:" + getRemainingCapacity()
                + ",trips_left:" + numDeliveriesBeforeMaintenance;

        String pilotName = getPilotName();
        if (pilotName.length() > 0) {
            droneDisplay += ",flown_by:" + pilotName;
        }

        return droneDisplay;
    }

    public static class DroneBuilder {
        private String droneID;
        private int weightCapacity;
        private int numDeliveriesBeforeMaintenance;
        private int numberOfOrders;
        private int totalOrderWeight;
        private String pilotFirstName;
        private String pilotLastName;
        private String waitTime;

        public Drone build() {
            return new Drone(this);
        }


        public DroneBuilder withDroneID(String droneID) {
            this.droneID = droneID;
            return this;
        }

        public DroneBuilder withWeightCapacity(int weightCapacity) {
            this.weightCapacity = weightCapacity;
            return this;
        }
        public DroneBuilder withNumDeliveriesBeforeMaintenance(int numDeliveriesBeforeMaintenance) {
            this.numDeliveriesBeforeMaintenance = numDeliveriesBeforeMaintenance;
            return this;
        }

        public DroneBuilder withNumberOfOrders(int numberOfOrders) {
            this.numberOfOrders = numberOfOrders;
            return this;
        }

        public DroneBuilder withPilotFirstName(String firstName) {
            this.pilotFirstName = firstName;
            return this;
        }

        public DroneBuilder withPilotLastName(String lastName) {
            this.pilotLastName = lastName;
            return this;
        }

        public DroneBuilder withTotalOrderWeight(int totalOrderWeight) {
            this.totalOrderWeight = totalOrderWeight;
            return this;
        }

        public DroneBuilder withWaitTime(String waitTime) {
            this.waitTime = waitTime;
            return this;
        }
    }
}