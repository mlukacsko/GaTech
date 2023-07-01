package org.gatech.dto;

import org.gatech.dbconnect.DatabaseEntity;

public class DronePilot extends DatabaseEntity {
    private String phoneNumber;
    private String firstName;
    private String lastName;
    private String taxId;
    private int monthsWorkedCount;
    private int salary;
    private String licenseId;
    private String accountId;
    private int successfulDeliveryCount;

    public DronePilot(String phoneNumber, String firstName, String lastName, String taxId, int monthsWorkedCount, int salary, String licenseId, String accountId, int successfulDeliveryCount) {
        this.phoneNumber = phoneNumber;
        this.firstName = firstName;
        this.lastName = lastName;
        this.taxId = taxId;
        this.monthsWorkedCount = monthsWorkedCount;
        this.salary = salary;
        this.licenseId = licenseId;
        this.accountId = accountId;
        this.successfulDeliveryCount = successfulDeliveryCount;
    }

    public DronePilot(DronePilotBuilder builder) {
        this.phoneNumber = builder.phoneNumber;
        this.firstName = builder.firstName;
        this.lastName = builder.lastName;
        this.taxId = builder.taxId;
        this.monthsWorkedCount = builder.monthsWorkedCount;
        this.salary = builder.salary;
        this.licenseId = builder.licenseId;
        this.accountId = builder.accountId;
        this.successfulDeliveryCount = builder.successfulDeliveryCount;
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

    public String getTaxId() {
        return taxId;
    }

    public void setTaxId(String taxId) {
        this.taxId = taxId;
    }

    public int getMonthsWorkedCount() {
        return monthsWorkedCount;
    }

    public void setMonthsWorkedCount(int monthsWorkedCount) {
        this.monthsWorkedCount = monthsWorkedCount;
    }

    public int getSalary() {
        return salary;
    }

    public void setSalary(int salary) {
        this.salary = salary;
    }

    public String getLicenseId() {
        return licenseId;
    }

    public void setLicenseId(String licenseId) {
        this.licenseId = licenseId;
    }

    public String getAccountId() {
        return accountId;
    }

    public void setAccountId(String accountId) {
        this.accountId = accountId;
    }

    public int getSuccessfulDeliveryCount() {
        return successfulDeliveryCount;
    }

    public void setSuccessfulDeliveryCount(int successfulDeliveryCount) {
        this.successfulDeliveryCount = successfulDeliveryCount;
    }

    @Override
    public String toString() {
        return "name:" + String.join("_", this.getFirstName(), this.getLastName()) +
                ",phone:" + this.getPhoneNumber() + ",taxID:" + this.getTaxId() +
                ",licenseID:" + licenseId + ",experience:" + successfulDeliveryCount;
    }

    public static class DronePilotBuilder {
        private String phoneNumber;
        private String firstName;
        private String lastName;
        private String taxId;
        private int monthsWorkedCount;
        private int salary;
        private String licenseId;
        private String accountId;
        private int successfulDeliveryCount;

        public DronePilot build() {
            return new DronePilot(this);
        }

        public DronePilotBuilder withPhoneNumber(String phoneNumber) {
            this.phoneNumber = phoneNumber;
            return this;
        }

        public DronePilotBuilder withFirstName(String firstName) {
            this.firstName = firstName;
            return this;
        }

        public DronePilotBuilder withLastName(String lastName) {
            this.lastName = lastName;
            return this;
        }

        public DronePilotBuilder withTaxId(String taxId) {
            this.taxId = taxId;
            return this;
        }

        public DronePilotBuilder withMonthsWorkedCount(int monthsWorkedCount) {
            this.monthsWorkedCount = monthsWorkedCount;
            return this;
        }

        public DronePilotBuilder withSalary(int salary) {
            this.salary = salary;
            return this;
        }

        public DronePilotBuilder withLicenseId(String licenseId) {
            this.licenseId = licenseId;
            return this;
        }

        public DronePilotBuilder withAccountId(String accountId) {
            this.accountId = accountId;
            return this;
        }

        public DronePilotBuilder withSuccessfulDeliveryCount(int successfulDeliveryCount) {
            this.successfulDeliveryCount = successfulDeliveryCount;
            return this;
        }

    }
}
