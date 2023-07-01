# Grocery Express App

## Overview

Grocery Express is a command line application that allows users or system administrators to manage deliveries of orders
from various stores to customers, by the use of drones flown by drone pilots. It was delivered as an in-memory command
line application in CS6310 as assignment 3.

Changes since this original version include:
* Using a database to persist the information entered in the system even after the command line is stopped.
* Implementing management of energy and corresponding changes to implement the aspects Distance, Time, and Cost.
  * Distance:
  * Time:
  * Cost:
* Implementing random distribution of coupons to customers over time 

In addition, command line help has been added: type `help` on the CLI for a listing of commands and their parameters.

### Data Storage

All commands from assignment 3 have been updated to read from, and write to, an SQL database (PostgreSQL). The original
65 CLI tests we were provided for assignment 3 are all passing against the new implementation.

### Energy Management + Distance/Time/Cost Aspects

Mode of operation:

* Distance is determined by the location of customers and stores. The locations are within a 2D coordinate system.
  To assign a location to a store or customer, the `set_customer_location` and `set_store_location` commands are used.
  To see customer and store locations, the `display_customer_location` and `display_store_location` commands are used.
  A customer and/or store may or may not have a location assigned to it.
  By default, customers and stores do NOT have locations.
* Time is managed via a database field that stores the current time. The `display_settings` command
  shows the current time. The `advance_time` CLI command changes the time to the desired time.
* Cost: A delivery fee is charged based on the hours of travel the drone has to do. It is 1 credit * the number of
  hours the delivery will take.

Energy is subtracted from drones when they deliver orders, and energy is increased when time is advanced.

* When a drone delivers an order, it determines the distance between the store and customer, and converts that into a
  number of hours to travel. The drone then loses energy according to its base energy consumption (which can be seen
  with `display_settings` and customized with `set_drone_energy_cost`) multiplied by the number of hours, but offset by
  energy gains within the hours it is set to travel in, which are percentages of the drones base energy consumption
  (`display_settings` shows these percentages, and `set_drone_energy_restoration` allows customizing them on an hour by
  hour basis). A drone that is traveling is unavailable to be assigned to an order or to deliver the order when it is
  purchased. For the drone to be usable again, you must `advance_time` to a point where the drone exceeds its travel
  duration.
* Energy is increased for all drones which are not delivering orders when `advance_time` is used according to the
  configured energy gain percentages. These gains are capped by the max energy capacity for drones, which can be seen
  with `display_settings` and customized with `set_drone_energy_capacity`.

Backwards compatibility:

* If either the customer or the store do not have a location, the order is delivered immediately and one hour of base
  energy cost is subtracted from the drone. The default base energy cost is 1 which matches how things worked when it
  was just the number of trips remaining.
* In addition, if either the customer or the store do not have a location, the drone does not accrue any travel time,
  so it is always available.
* In addition, if either the customer or the store does not have a location, the delivery fee is brought down to zero.

### Coupons
Coupons are distributed over time based on the frequency the user provides. The user will be able to control this by 
using the distribute_frequency command.

* distribute_frequency: takes in two arguments, <num_of_commands> and <num_coupons>.
* Once this command is executed, the system will distribute <num_coupons> coupons after every <num_of_commands> commands
are executed.
* (E.g.) distribute_frequency,3,8 - this command will tell the system to distribute eight coupons after every three commands are executed.

In order to apply the coupons, the purchase_order command will now take in an optional argument flag, <apply_coupon>,
this value is either true or false. By default, the coupons will not be applied to the order.

* purchase_order: takes in two required arguments, <store_name> and <order_identifier>, with an additional optional argument, <apply_coupon>
* The original purchase_order command will still work with the original arguments.
* (E.g.) purchase_order,kroger,purchaseA,true will purchase the order and apply the coupons the customer has to the order total cost.



## Using the Application

NOTE: these are instructions for using the application as a user or system administrator. Developer instructions are in
the following section.

To start the application, run the following command:

`docker-compose up`

The command line interface is run as a set of independent docker containers that can be connected to via `docker attach`.
Think of the system as having multiple heads that are available to connect to. A user or system administrator will
connect to an available CLI as such:

`docker attach grocery-express-cli-{number}`,

where `{number}` is the specific replica number of the CLI to attach to, e.g `1`, `2`, etc. The default system
configuration is to have two CLI replicas, but this number can be easily changed in the `docker-compose.yml`
configuration.

When they have finished using the CLI, the user or system administrator issues the `stop` command. Another person can
now use this CLI container for their purposes. Multiple persons connecting to the same CLI replica simultaneously is not
a supported use case, though it does work, the result is that both persons will see the terminal outputs corresponding
to each other's usage of the system.

## Run CLI Acceptance Tests

1. `docker-compose up` to start the application.
2. In another terminal, run `sh scripts/batch-running.sh` to run all the original 65 CLI tests, or
   `sh scripts/test-running.sh [scenario_id]` to run a single scenario.

NOTE: There are two additional scenarios:
  * The test with `scenario_id` of 99, that tests the energy management and handling of the new aspects of
    distance/time/cost.
  * The test with `scenario_id` of 98, that tests the coupons distribution.

## Development

### Requirements

* JDK 17

### Setup

1. Always make sure to have a database running. By using `docker-compose up db` you will have a database that the
   application will know how to connect to even in development mode or via the JAR executable.

### Option A: Run the application in normal or debug mode

1. Run `./gradlew app:run` to run the application. You can set this as a run configuration in an IDE to run and/or
   debug the application.

### Building the JAR executables

1. Run `./gradlew build` to build the executable JAR.
2. Run the JAR file using `java -jar app/build/libs/app.jar`.