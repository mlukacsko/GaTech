package org.gatech.dto;

import org.gatech.dbconnect.DatabaseEntity;

import java.util.ArrayList;
import java.util.List;

public class Order extends DatabaseEntity {

    private String ID;
    private int storeID;
    private String orderID;
    private int droneID;
    private String customerID;
    private String storeName;
    private List<OrderItem> items;


    public Order(String storeName, String ID, int storeID, String orderID , int droneID, String customerID) {
        this.storeName = storeName;
        this.ID = ID;
        this.storeID = storeID;
        this.orderID = orderID;
        this.droneID = droneID;
        this.customerID = customerID;
        this.items = new ArrayList<>();
    }

    private Order(OrderBuilder builder) {
        this.ID = builder.ID;
        this.storeID = builder.storeID;
        this.orderID = builder.orderID;
        this.droneID = builder.droneID;
        this.customerID = builder.customerID;
        this.items = new ArrayList<>();
    }

    public String getStoreName() {return storeName;}

    public void setStoreName(String storeName) {
        this.storeName = storeName;
    }
    public String getID() {
        return ID;
    }
    public void setID(String ID) {
        this.ID = ID;
    }
    public int getStoreID() {
        return storeID;
    }
    public void setStoreID(int storeID ) {
        this.storeID = storeID;
    }
    public String getOrderID() {return orderID; }
    public void getOrderID(String orderID ) { this.orderID = orderID; }
    public int getDroneID() { return droneID;}
    public void setDroneID(int droneID ) {
        this.droneID = droneID;
    }
    public String getCustomerID() {return customerID;}
    public void setCustomerID(String customerID ) {
        this.customerID = customerID;
    }
    public List<OrderItem> getItems() {
        return items;
    }
    public void setItems(List<OrderItem> items) {
        this.items = items;
    }

    @Override
    public String toString() {
        String orderList = "orderID:" + orderID;
        if (items.isEmpty()){
            return orderList;
        }
        else
            for (OrderItem item : items){
                orderList += "\nitem_name:" + item.getItemName() +
                        ",total_quantity:" + item.getTotalQuantity() +
                        ",total_cost:" + item.getTotalCost() + ",total_weight:" + item.getTotalWeight();

            }
        return orderList;
    }

    public static class OrderBuilder {
        private String ID;
        private int storeID;
        private String orderID;
        private int droneID;
        private String customerID;

        private Order order;


        public Order build() {
            return new Order(this);
        }

        public OrderBuilder withID(String ID) {
            this.ID = ID;
            return this;
        }

        public OrderBuilder withStoreID(int storeID) {
            this.storeID = storeID;
            return this;
        }
        public OrderBuilder withOrderID(String orderID) {
            this.orderID = orderID;
            return this;
        }
        public OrderBuilder withDroneID(int droneID ) {
            this.droneID = droneID;
            return this;
        }
        public OrderBuilder withCustomerID(String customerID ) {
            this.customerID = customerID;
            return this;
        }
    }
}
