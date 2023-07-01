package groceryexpress;

import com.google.inject.Inject;

import java.util.Optional;
import java.util.Scanner;

public class CLI {

    private final DeliveryService deliveryService;

    public CLI(DeliveryService deliveryService) {
        this.deliveryService = deliveryService;
    }

    public void commandLoop() {
        Scanner commandLineInput = new Scanner(System.in);
        String wholeInputLine;
        String[] tokens;
        final String DELIMITER = ",";

        while (true) {
            try {
                // Determine the next command and echo it to the monitor for testing purposes
                wholeInputLine = commandLineInput.nextLine();
                tokens = wholeInputLine.split(DELIMITER);
                System.out.println("> " + wholeInputLine);

                if (tokens[0].equals("make_store")) {
                    this.deliveryService.handleMakeStore(tokens[1], tokens[2]);
                } else if (tokens[0].equals("display_stores")) {
                    this.deliveryService.handleDisplayStores();
                } else if (tokens[0].equals("sell_item")) {
                    this.deliveryService.handleSellItem(tokens[1], tokens[2], tokens[3]);
                } else if (tokens[0].equals("display_items")) {
                    this.deliveryService.handleDisplayItems(tokens[1]);
                } else if (tokens[0].equals("make_pilot")) {
                    this.deliveryService.handleMakePilot(tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6], tokens[7]);
                } else if (tokens[0].equals("display_pilots")) {
                    this.deliveryService.handleDisplayPilots();
                } else if (tokens[0].equals("make_drone")) {
                    this.deliveryService.handleMakeDrone(tokens[1], tokens[2], tokens[3], tokens[4]);
                } else if (tokens[0].equals("display_drones")) {
                    this.deliveryService.handleDisplayDrones(tokens[1]);
                } else if (tokens[0].equals("fly_drone")) {
                    this.deliveryService.handleFlyDrone(tokens[1], tokens[2], tokens[3]);
                } else if (tokens[0].equals("make_customer")) {
                    this.deliveryService.handleMakeCustomer(tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6]);
                } else if (tokens[0].equals("display_customers")) {
                    this.deliveryService.handleDisplayCustomers();
                } else if (tokens[0].equals("start_order")) {
                    this.deliveryService.handleStartOrder(tokens[1], tokens[2], tokens[3], tokens[4]);
                } else if (tokens[0].equals("display_orders")) {
                    this.deliveryService.handleDisplayOrders(tokens[1]);
                } else if (tokens[0].equals("request_item")) {
                    this.deliveryService.handleRequestItem(tokens[1], tokens[2], tokens[3], tokens[4], tokens[5]);
                } else if (tokens[0].equals("purchase_order")) {
                    if (tokens.length == 3) {  // if there is no apply coupons
                        this.deliveryService.handlePurchaseOrder(tokens[1], tokens[2], Optional.empty());
                    } else {
                        this.deliveryService.handlePurchaseOrder(tokens[1], tokens[2], Optional.of(Boolean.parseBoolean(tokens[3])));
                    }
                } else if (tokens[0].equals("cancel_order")) {
                    this.deliveryService.handleCancelOrder(tokens[1], tokens[2]);
                } else if (tokens[0].equals("transfer_order")) {
                    this.deliveryService.handleTransferOrder(tokens[1], tokens[2], tokens[3]);
                } else if (tokens[0].equals("display_efficiency")) {
                    this.deliveryService.handleDisplayEfficiency();
                } else if (tokens[0].equals("distribute_frequency")) {
                    this.deliveryService.handleDistributeFrequency(Integer.parseInt(tokens[1]), Integer.parseInt(tokens[2]));
                } else if (tokens[0].equals("display_settings")) {
                    this.deliveryService.displaySettings();
                } else if (tokens[0].equals("wait_until")) {
                    this.deliveryService.handleWaitUntil(tokens[1]);
                } else if (tokens[0].equals("set_drone_energy_cost")) {
                    this.deliveryService.setDroneEnergyCost(tokens[1]);
                } else if (tokens[0].equals("set_drone_energy_capacity")) {
                    this.deliveryService.setDroneEnergyCapacity(tokens[1]);
                } else if (tokens[0].equals("set_drone_energy_restoration")) {
                    this.deliveryService.setDroneEnergyRestoration(tokens[1], tokens[2]);
                } else if (tokens[0].equals("display_store_location")) {
                    this.deliveryService.handleDisplayStoreLocation(tokens[1]);
                } else if (tokens[0].equals("display_customer_location")) {
                    this.deliveryService.handleDisplayCustomerLocation(tokens[1]);
                } else if (tokens[0].equals("set_store_location")) {
                    this.deliveryService.setStoreLocation(tokens[1], tokens[2], tokens[3]);
                } else if (tokens[0].equals("set_customer_location")) {
                    this.deliveryService.setCustomerLocation(tokens[1], tokens[2], tokens[3]);
                } else if (tokens[0].equals("help")) {
                    this.deliveryService.handleHelp();
                } else if (tokens[0].equals("stop")) {
                    System.out.println("stop acknowledged");
                    break;
                } else {
                    // ignore
                }
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println();
            }
        }

        System.out.println("simulation terminated");
        commandLineInput.close();
    }
}
