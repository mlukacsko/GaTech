package org.gatech.dto;

import org.gatech.dbconnect.DatabaseEntity;

public class Store extends DatabaseEntity {

    private String name;
    private int earnedRevenue;
    private int completedOrderCount;
    private int transferredOrderCount;
    private int droneOverloadCount;

    public Store(String name, int earnedRevenue, int completedOrderCount, int transferredOrderCount, int droneOverloadCount) {
        this.name = name;
        this.earnedRevenue = earnedRevenue;
        this.completedOrderCount = completedOrderCount;
        this.transferredOrderCount = transferredOrderCount;
        this.droneOverloadCount = droneOverloadCount;
    }

    private Store(StoreBuilder builder) {
        this.name = builder.name;
        this.earnedRevenue = builder.earnedRevenue;
        this.completedOrderCount = builder.completedOrderCount;
        this.transferredOrderCount = builder.transferredOrderCount;
        this.droneOverloadCount = builder.droneOverloadCount;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getEarnedRevenue() {
        return earnedRevenue;
    }

    public void setEarnedRevenue(int earnedRevenue) {
        this.earnedRevenue = earnedRevenue;
    }

    public int getCompletedOrderCount() {
        return completedOrderCount;
    }

    public void setCompletedOrderCount(int completedOrderCount) {
        this.completedOrderCount = completedOrderCount;
    }

    public int getTransferredOrderCount() {
        return transferredOrderCount;
    }

    public void setTransferredOrderCount(int transferredOrderCount) {
        this.transferredOrderCount = transferredOrderCount;
    }

    public int getDroneOverloadCount() {
        return droneOverloadCount;
    }

    public void setDroneOverloadCount(int droneOverloadCount) {
        this.droneOverloadCount = droneOverloadCount;
    }

    public void displayEfficiency() {
        System.out.println("name:" + name + ",purchases:" + completedOrderCount + ",overloads:" + droneOverloadCount + ",transfers:" + transferredOrderCount);
    }

    @Override
    public String toString() {
        return "name:" + name + ",revenue:" + earnedRevenue;
    }

    public static class StoreBuilder {
        private String name;
        private int earnedRevenue;
        private int completedOrderCount;
        private int transferredOrderCount;
        private int droneOverloadCount;

        public Store build() {
            return new Store(this);
        }

        public StoreBuilder withName(String name) {
            this.name = name;
            return this;
        }

        public StoreBuilder withEarnedRevenue(int earnedRevenue) {
            this.earnedRevenue = earnedRevenue;
            return this;
        }

        public StoreBuilder withCompletedOrderCount(int completedOrderCount) {
            this.completedOrderCount = completedOrderCount;
            return this;
        }

        public StoreBuilder withTransferredOrderCount(int transferredOrderCount) {
            this.transferredOrderCount = transferredOrderCount;
            return this;
        }

        public StoreBuilder withDroneOverloadCount(int droneOverloadCount) {
            this.droneOverloadCount = droneOverloadCount;
            return this;
        }
    }
}
