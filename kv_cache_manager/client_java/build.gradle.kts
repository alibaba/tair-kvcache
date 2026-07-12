plugins {
    `java-library`
    id("com.google.protobuf") version "0.9.4"
}

group = "com.alibaba.tair.kvcm"
version = "0.1.0-SNAPSHOT"

repositories {
    maven {
        url = uri("https://maven.aliyun.com/repository/central")
    }
    mavenCentral()
}

val grpcVersion = "1.58.0"
val protobufVersion = "3.25.3"
val okhttpVersion = "4.12.0"

java {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
}

tasks.withType<JavaCompile> {
    options.encoding = "UTF-8"
}

dependencies {
    // gRPC (api: generated stubs expose GeneratedMessageV3 / MessageOrBuilder to consumers)
    api("io.grpc:grpc-protobuf:${grpcVersion}")
    api("io.grpc:grpc-stub:${grpcVersion}")
    implementation("io.grpc:grpc-netty-shaded:${grpcVersion}")

    // Protobuf (api: generated messages expose protobuf supertypes/builders to consumers)
    api("com.google.protobuf:protobuf-java:${protobufVersion}")
    api("com.google.protobuf:protobuf-java-util:${protobufVersion}")

    // javax.annotation for @Generated (required by gRPC stubs)
    compileOnly("javax.annotation:javax.annotation-api:1.3.2")

    // HTTP fallback: OkHttp
    implementation("com.squareup.okhttp3:okhttp:${okhttpVersion}")

    // SLF4J for logging
    implementation("org.slf4j:slf4j-api:2.0.13")

    // Test
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.3")
    testImplementation("io.grpc:grpc-testing:${grpcVersion}")
    testImplementation("io.grpc:grpc-inprocess:${grpcVersion}")
    testImplementation("com.squareup.okhttp3:mockwebserver:${okhttpVersion}")
    testImplementation("org.slf4j:slf4j-simple:2.0.13")
}

protobuf {
    protoc {
        artifact = "com.google.protobuf:protoc:${protobufVersion}"
    }
    plugins {
        create("grpc") {
            artifact = "io.grpc:protoc-gen-grpc-java:${grpcVersion}"
        }
    }
    generateProtoTasks {
        all().forEach { task ->
            task.plugins {
                create("grpc")
            }
        }
    }
}

sourceSets {
    main {
        proto {
            srcDir("proto")
        }
        java {
            srcDirs(
                "build/generated/source/proto/main/java",
                "build/generated/source/proto/main/grpc"
            )
        }
    }
    create("integrationTest") {
        compileClasspath += sourceSets.main.get().output
        runtimeClasspath += sourceSets.main.get().output
    }
}

val integrationTestImplementation by configurations.getting {
    extendsFrom(configurations.testImplementation.get())
}

val integrationTestRuntimeOnly by configurations.getting {
    extendsFrom(configurations.testRuntimeOnly.get())
}

tasks.test {
    useJUnitPlatform()
}

tasks.register<Test>("integrationTest") {
    description = "Runs integration tests against a real KVCM server."
    group = "verification"
    testClassesDirs = sourceSets["integrationTest"].output.classesDirs
    classpath = sourceSets["integrationTest"].runtimeClasspath
    useJUnitPlatform()
    mustRunAfter(tasks.test)
    
    // Pass KVCM_BIN environment variable to tests (only if set)
    val kvcmBin = System.getenv("KVCM_BIN")
    if (kvcmBin != null) {
        environment("KVCM_BIN", kvcmBin)
    }
    
    // Show test output
    testLogging {
        events("passed", "skipped", "failed", "standardOut", "standardError")
    }
}

tasks.check {
    dependsOn("integrationTest")
}
