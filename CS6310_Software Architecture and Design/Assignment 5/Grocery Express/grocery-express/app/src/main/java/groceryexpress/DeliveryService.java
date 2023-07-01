package groceryexpress;

import com.google.inject.Inject;
import org.gatech.dto.*;
import org.gatech.exception.GroceryExpressException;
import org.gatech.service.customer.CustomerService;
import org.gatech.service.location.LocationService;
import org.gatech.service.settings.SettingsService;
import org.gatech.service.pilot.DronePilotService;
import org.gatech.service.store.StoreService;


import java.sql.SQLException;
import java.util.Optional;
import java.util.List;

public final class DeliveryService {

    private final StoreService storeService;
    private final CustomerService customerService;
    private final DronePilotService dronePilotService;
    private final SettingsService settingsService;
    private final LocationService locationService;

    private int commandCounter;
    private int frequency;
    private int numberOfCoupons;

    @Inject
    public DeliveryService(
            StoreService storeService,
            CustomerService customerService,
            DronePilotService dronePilotService,
            SettingsService settingsService,
            LocationService locationService
    ) {
        this.storeService = storeService;
        this.customerService = customerService;
        this.dronePilotService = dronePilotService;
        this.commandCounter = 0;
        this.frequency = Integer.MAX_VALUE;  // set to max value until an actual value is set
        this.numberOfCoupons = 0;
        this.settingsService = settingsService;
        this.locationService = locationService;
    }

    public void handleMakeStore(String storeName, String revenue) throws SQLException, ClassNotFoundException {
        try {
            storeService.createStore(
                    new Store.StoreBuilder()
                            .withName(storeName)
                            .withEarnedRevenue(Integer.parseInt(revenue))
                            .withCompletedOrderCount(0)
                            .withTransferredOrderCount(0)
                            .withDroneOverloadCount(0)
                            .build()
            );
            System.out.println("OK:change_completed");
        } catch (GroceryExpressException exception) {
            System.out.println(exception.getMessage());
        } catch (Exception exception) {
            System.out.println("An uncaught exception occurred");
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void handleDisplayStores() throws SQLException, ClassNotFoundException {
        storeService.getStores().forEach(System.out::println);
        System.out.println("OK:display_completed");
        ++commandCounter;
        distributeCoupons();
    }

    public void handleSellItem(String storeName, String itemName, String itemWeight) throws SQLException, ClassNotFoundException {
        try {
            storeService.sellItem(
                    new Item.ItemBuilder()
                            .withStoreName(storeName)
                            .withItemName(itemName)
                            .withItemWeight(Integer.parseInt(itemWeight))
                            .build(),
                    storeName
            );
            System.out.println("OK:change_completed");
        } catch (GroceryExpressException e) {
            System.out.println(e.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void handleDisplayItems(String storeName) throws SQLException, ClassNotFoundException {
        try {
            storeService.getItems(storeName).forEach(System.out::println);
            System.out.println("OK:display_completed");
        } catch (GroceryExpressException e) {
            System.out.println(e.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void handleMakePilot(
            String accountId,
            String firstName,
            String lastName,
            String phoneNumber,
            String taxIdentifier,
            String licenseId,
            String numSuccessfulDeliveries
    ) throws SQLException, ClassNotFoundException {
        try {
            dronePilotService.createDronePilot(
                    new DronePilot.DronePilotBuilder()
                            .withAccountId(accountId)
                            .withFirstName(firstName)
                            .withLastName(lastName)
                            .withPhoneNumber(phoneNumber)
                            .withTaxId(taxIdentifier)
                            .withLicenseId(licenseId)
                            .withSuccessfulDeliveryCount(Integer.parseInt(numSuccessfulDeliveries))
                            .withSalary(0)
                            .withMonthsWorkedCount(0)
                            .build()
            );
            System.out.println("OK:change_completed");
        } catch (GroceryExpressException exception) {
            System.out.println(exception.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }

    }

    public void handleDisplayPilots() throws SQLException, ClassNotFoundException {
        dronePilotService.getDronePilots().forEach(System.out::println);
        System.out.println("OK:display_completed");
        ++commandCounter;
        distributeCoupons();
    }

    public void handleMakeDrone(String storeName, String droneId, String weightCapacity, String numDeliveriesBeforeMaintenance) throws SQLException, ClassNotFoundException, GroceryExpressException {
        try {
            storeService.makeDrone(
                    storeName,
                    new Drone.DroneBuilder()
                            .withDroneID(droneId)
                            .withWeightCapacity(Integer.parseInt(weightCapacity))
                            .withNumDeliveriesBeforeMaintenance(Integer.parseInt(numDeliveriesBeforeMaintenance))
                            .build()
            );
            System.out.println("OK:change_completed");
        } catch (GroceryExpressException exception) {
            System.out.println(exception.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void handleDisplayDrones(String storeName) throws SQLException, ClassNotFoundException, GroceryExpressException {
        try {
            storeService.getDrones(storeName).forEach(System.out::println);
            System.out.println("OK:display_completed");
        } catch (GroceryExpressException e) {
            System.out.println(e.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }

    }

    public void handleFlyDrone(String storeName, String droneId, String pilotAccountId) throws SQLException, ClassNotFoundException {
        try {
            storeService.flyDrone(storeName, droneId, pilotAccountId);
            System.out.println("OK:change_completed");
        }
        catch (GroceryExpressException e) {
            System.out.println(e.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void handleMakeCustomer(
            String accountId,
            String firstName,
            String lastName,
            String phoneNumber,
            String rating,
            String credits
    ) throws SQLException, ClassNotFoundException {
        try {
            customerService.createCustomer(
                    new Customer.CustomerBuilder()
                            .withAccountId(accountId)
                            .withFirstName(firstName)
                            .withLastName(lastName)
                            .withPhoneNumber(phoneNumber)
                            .withRating(Integer.parseInt(rating))
                            .withCredits(Integer.parseInt(credits))
                            .build()
            );
            System.out.println("OK:change_completed");
        } catch (GroceryExpressException e) {
            System.out.println(e.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void handleDisplayCustomers() throws SQLException, ClassNotFoundException {
        customerService.getCustomers().forEach(System.out::println);
        System.out.println("OK:display_completed");
        ++commandCounter;
        distributeCoupons();
    }

    public void handleStartOrder(String storeName, String orderId, String droneId, String customerAccountId) throws SQLException, ClassNotFoundException{
        try {
            storeService.makeOrder(
                    storeName,
                    new Order.OrderBuilder()
                            .withOrderID(orderId)
                            .withDroneID(Integer.parseInt(droneId))
                            .withCustomerID(customerAccountId)
                            .build()
            );
            System.out.println("OK:change_completed");
        }
        catch(GroceryExpressException exception) {
            System.out.println(exception.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void handleDisplayOrders(String storeName) throws SQLException, ClassNotFoundException {
        try {
            storeService.getOrder(storeName).forEach(System.out::println);
            System.out.println("OK:display_completed");
        }
        catch (GroceryExpressException e){
            System.out.println(e.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void handleRequestItem(String storeName, String orderId, String itemName, String quantity, String unitPrice) throws SQLException, ClassNotFoundException {
        try {
            storeService.requestItem(storeName, orderId, itemName, Integer.parseInt(quantity), Integer.parseInt(unitPrice));
            System.out.println("OK:change_completed");
        } catch (GroceryExpressException e) {
            System.out.println(e.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void handlePurchaseOrder(String storeName, String orderID, Optional<Boolean> applyCoupon) throws SQLException, ClassNotFoundException {
        try {
            storeService.purchaseOrder(storeName, orderID, applyCoupon);
            System.out.println("OK:change_completed");
        }
        catch (GroceryExpressException e) {
            System.out.println(e.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void handleCancelOrder(String storeName, String orderId) throws SQLException, ClassNotFoundException{
        try {
            storeService.cancelOrder(storeName, orderId);
            System.out.println("OK:change_completed");
        }
        catch (GroceryExpressException exception) {
            System.out.println(exception.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void handleTransferOrder(String storeName, String orderId, String droneId) throws SQLException, ClassNotFoundException {
        try {
            storeService.transferOrder(storeName, orderId, droneId);
            System.out.println("OK:change_completed");
        } catch (GroceryExpressException exception) {
            System.out.println(exception.getMessage());
        } finally {
            ++commandCounter;
            distributeCoupons();
        }
    }

    public void displaySettings() throws SQLException, ClassNotFoundException {
        Settings settings = settingsService.getSettings().get(0);
        System.out.println(settings.toString());
        System.out.println("OK:display_completed");
    }

    public void handleWaitUntil(String hour) {
        try {
            settingsService.advanceTime(hour);
        } catch (Exception exception) {
            System.out.println("An unknown error occurred.");
        }
    }

    public void setDroneEnergyCost(String energyCost) {
        try {
            settingsService.setDroneEnergyCost(energyCost);
        } catch (Exception exception) {
            System.out.println("An unknown error occurred.");
        }
    }

    public void setDroneEnergyCapacity(String energyCapacity) {
        try {
            settingsService.setDroneEnergyCapacity(energyCapacity);
        } catch (Exception exception) {
            System.out.println("An unknown error occurred");
        }
    }

    public void setDroneEnergyRestoration(String time, String energyRestoration) {
        try {
            settingsService.setDroneEnergyRestoration(time, energyRestoration);
        } catch (Exception exception) {
            System.out.println("An unknown error occurred.");
        }
    }

    public void handleDisplayCustomerLocation(String customerAccountId) throws SQLException, ClassNotFoundException {
        List<Location> locations = this.locationService.getCustomerLocation(customerAccountId);
        locations.forEach(System.out::println);
        System.out.println("OK:display_completed");
    }

    public void handleDisplayStoreLocation(String storeName) throws SQLException, ClassNotFoundException {
        List<Location> locations = this.locationService.getStoreLocation(storeName);
        locations.forEach(System.out::println);
        System.out.println("OK:display_completed");
    }

    public void setStoreLocation(String storeName, String xCoordinate, String yCoordinate) {
        try {
            settingsService.setStoreLocation(storeName, xCoordinate, yCoordinate);
        } catch (Exception exception) {
            System.out.println("An unknown error occurred.");
        }
    }

    public void setCustomerLocation(String customerAccountId, String xCoordinate, String yCoordinate) {
        try {
            settingsService.setCustomerLocation(customerAccountId, xCoordinate, yCoordinate);
        } catch (Exception exception) {
            System.out.println("An unknown error occurred.");
        }
    }

    public void handleHelp() {
        System.out.println("Available commands:");
        System.out.println("-----------------");
        System.out.println("make_store,<store-name>,<initial-revenue>");
        System.out.println("display_stores");
        System.out.println("sell_item,<store-name>,<item-name>,<item-weight>");
        System.out.println("display_items,<store-name>");
        System.out.println("make_pilot,<account-id>,<first-name>,<last-name>,<phone>,<tax-id>,<license-id>,<num-successful-deliveries>");
        System.out.println("display_pilots");
        System.out.println("make_drone,<store-name>,<drone-id>,<weight-capacity>,<deliveries-before-maintenance>");
        System.out.println("display_drones,<store-name>");
        System.out.println("fly_drone,<store-name>,<drone-id>,<pilot-account-id>");
        System.out.println("make_customer,<account-id>,<first-name>,<last-name>,<phone>,<rating>,<credits>");
        System.out.println("display_customers");
        System.out.println("start_order,<store-name>,<order-id>,<drone-id>,<customer-account-id>");
        System.out.println("display_orders,<store-name>");
        System.out.println("request_item,<store-name>,<order-id>,<item-name>,<item-quantity>,<item-price>");
        System.out.println("purchase_order,<store-name>,<order-id>,[use-coupons-flag]");
        System.out.println("cancel_order,<store-name>,<order-id>");
        System.out.println("transfer_order,<store-name>,<order-id>,<drone-id>");
        System.out.println("display_efficiency");
        System.out.println("display_settings");
        System.out.println("wait_until,<hour>");
        System.out.println("set_drone_energy_cost,<amount>");
        System.out.println("set_drone_energy_capacity,<amount>");
        System.out.println("set_drone_energy_restoration,<time>,<amount>");
        System.out.println("display_customer_location,<account-id>");
        System.out.println("display_store_location,<store-name>");
        System.out.println("set_store_location,<store-name>,<x-coordinate>,<y-coordinate>");
        System.out.println("set_customer_location,<account-id>,<x-coordinate>,<y-coordinate>");
        System.out.println("distribute_frequency,<frequency>,<num-coupons>");
    }

    public void handleDisplayEfficiency() throws SQLException, ClassNotFoundException {
        storeService.getStores().forEach(Store::displayEfficiency);
        System.out.println("OK:display_completed");
        ++commandCounter;
        distributeCoupons();
    }

    public void handleDistributeFrequency(int frequency, int numberOfCoupons) {
        this.frequency = frequency;
        this.numberOfCoupons = numberOfCoupons;
        commandCounter = 0;
        System.out.println("OK:change_completed");
    }

    private void distributeCoupons() throws SQLException, ClassNotFoundException {
        if (commandCounter >= frequency) {
            customerService.distributeCoupons(numberOfCoupons);
            commandCounter = 0;
            System.out.println("ALERT:coupons_distributed");
        }
    }
}
