package org.gatech.dto;

import org.gatech.dbconnect.DatabaseEntity;

public class OrderItem extends DatabaseEntity {

    private String itemName;
    private int totalQuantity;
    private int totalCost;
    private int totalWeight;

    public OrderItem(String itemName, int totalQuantity, int totalCost, int totalWeight) {
        this.itemName = itemName;
        this.totalQuantity = totalQuantity;
        this.totalCost = totalCost;
        this.totalWeight = totalWeight;
    }

    private OrderItem(OrderItemBuilder builder) {
        this.itemName = builder.itemName;
        this.totalQuantity = builder.totalQuantity;
        this.totalCost = builder.totalCost;
        this.totalWeight = builder.totalWeight;
    }

    public String getItemName() {
        return itemName;
    }

    public void setItemName(String itemName) {
        this.itemName = itemName;
    }

    public int getTotalQuantity() {
        return totalQuantity;
    }

    public void setTotalQuantity(int totalQuantity) {
        this.totalQuantity = totalQuantity;
    }

    public int getTotalCost() {
        return totalCost;
    }

    public void setTotalCost(int totalCost) {
        this.totalCost = totalCost;
    }

    public int getTotalWeight() {
        return totalWeight;
    }

    public void setTotalWeight(int totalWeight) {
        this.totalWeight = totalWeight;
    }

    public static class OrderItemBuilder {
        private String itemName;
        private int totalQuantity;
        private int totalCost;
        private int totalWeight;

        public OrderItem build() {
            return new OrderItem(this);
        }

        public OrderItemBuilder withItemName(String itemName) {
            this.itemName = itemName;
            return this;
        }

        public OrderItemBuilder withTotalQuantity(int totalQuantity) {
            this.totalQuantity = totalQuantity;
            return this;
        }

        public OrderItemBuilder withTotalCost(int totalCost) {
            this.totalCost = totalCost;
            return this;
        }

        public OrderItemBuilder withTotalWeight(int totalWeight) {
            this.totalWeight = totalWeight;
            return this;
        }

    }
}
