# Compile the project with gradle
FROM gradle:7.5.1-jdk17 AS builder

COPY app/src /usr/src/app/src
COPY shared/src /usr/src/shared/src

COPY app/build.gradle /usr/src/app/build.gradle
COPY shared/build.gradle /usr/src/shared/build.gradle

COPY gradle /usr/src/gradle
COPY settings.gradle /usr/src/settings.gradle
COPY gradlew /usr/src/gradlew

WORKDIR /usr/src
RUN gradle :shared:publish
RUN gradle build

# Copy the JAR and test scenarios into our final image
FROM openjdk:17-slim
WORKDIR /usr/src
COPY app/src/test/cli/inputs ./
COPY app/src/test/cli/expected ./
COPY --from=builder /usr/src/app/build/libs/app.jar ./app.jar
