package org.gatech.dto;

import org.gatech.dbconnect.DatabaseEntity;

public class Customer extends DatabaseEntity {

    private String phoneNumber;
    private String firstName;
    private String lastName;
    private String accountId;
    private int credits;
    private int rating;

    public Customer(String phoneNumber, String firstName, String lastName, String accountId, int credits, int rating) {
        this.phoneNumber = phoneNumber;
        this.firstName = firstName;
        this.lastName = lastName;
        this.accountId = accountId;
        this.credits = credits;
        this.rating = rating;
    }

    private Customer(CustomerBuilder builder) {
        this.phoneNumber = builder.phoneNumber;
        this.firstName = builder.firstName;
        this.lastName = builder.lastName;
        this.accountId = builder.accountId;
        this.credits = builder.credits;
        this.rating = builder.rating;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getAccountId() {
        return accountId;
    }

    public void setAccountId(String accountId) {
        this.accountId = accountId;
    }

    public int getCredits() {
        return credits;
    }

    public void setCredits(int credits) {
        this.credits = credits;
    }

    public int getRating() {
        return rating;
    }

    public void setRating(int rating) {
        this.rating = rating;
    }

    @Override
    public String toString() {
        return "name:" + String.join("_", this.getFirstName(), this.getLastName()) +
                ",phone:" + this.getPhoneNumber() + ",rating:" + rating + ",credit:" + credits;
    }

    public static class CustomerBuilder {
        private String phoneNumber;
        private String firstName;
        private String lastName;
        private String accountId;
        private int credits;
        private int rating;

        public Customer build() {
            return new Customer(this);
        }

        public CustomerBuilder withPhoneNumber(String phoneNumber) {
            this.phoneNumber = phoneNumber;
            return this;
        }

        public CustomerBuilder withFirstName(String firstName) {
            this.firstName = firstName;
            return this;
        }

        public CustomerBuilder withLastName(String lastName) {
            this.lastName = lastName;
            return this;
        }

        public CustomerBuilder withAccountId(String accountId) {
            this.accountId = accountId;
            return this;
        }

        public CustomerBuilder withCredits(int credits) {
            this.credits = credits;
            return this;
        }

        public CustomerBuilder withRating(int rating) {
            this.rating = rating;
            return this;
        }

    }
}