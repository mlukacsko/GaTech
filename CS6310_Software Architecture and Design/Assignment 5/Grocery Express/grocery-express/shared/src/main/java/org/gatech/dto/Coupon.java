package org.gatech.dto;

import org.gatech.dbconnect.DatabaseEntity;

import java.util.Date;

public class Coupon extends DatabaseEntity {
    private int customerId;
    private int percentage;
    private Date expirationDate;

    public Coupon(int customerId, int percentage, Date expirationDate) {
        this.customerId = customerId;
        this.percentage = percentage;
        this.expirationDate = expirationDate;
    }

    private Coupon(CouponBuilder builder) {
        this.customerId = builder.customerId;
        this.percentage = builder.percentage;
        this.expirationDate = builder.expirationDate;
    }

    public int getCustomerId() {
        return customerId;
    }

    public void setCustomerId(int customerId) {
        this.customerId = customerId;
    }

    public int getPercentage() {
        return percentage;
    }

    public void setPercentage(int percentage) {
        this.percentage = percentage;
    }

    public Date getExpirationDate() {
        return expirationDate;
    }

    public void setExpirationDate(Date expirationDate) {
        this.expirationDate = expirationDate;
    }

    public static class CouponBuilder {
        private int customerId;
        private int percentage;
        private Date expirationDate;

        public Coupon build() {
            return new Coupon(this);
        }

        public CouponBuilder withCustomerId(int customerId) {
            this.customerId = customerId;
            return this;
        }

        public CouponBuilder withPercentage(int percentage) {
            this.percentage = percentage;
            return this;
        }

        public CouponBuilder withExpirationDate(Date expirationDate) {
            this.expirationDate = expirationDate;
            return this;
        }
    }
}
