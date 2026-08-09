plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
    application
}

dependencies {
    implementation(project(":core-contracts"))
    implementation(project(":core-motion"))
    implementation(project(":core-alignment"))
    implementation(project(":sync-v2-integration"))
    implementation(libs.kotlinx.serialization.json)
    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(21)
    compilerOptions {
        allWarningsAsErrors.set(true)
    }
}

application {
    mainClass.set("ai.senp.sync.validation.MainKt")
}

tasks.test {
    useJUnitPlatform()
}

sourceSets {
    main {
        resources.srcDir(rootProject.file("core-contracts/src/test/resources"))
    }
}

tasks.named<JavaExec>("run") {
    workingDir(rootProject.projectDir)
}
