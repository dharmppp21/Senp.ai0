plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(21)
    compilerOptions {
        allWarningsAsErrors.set(true)
    }
}

dependencies {
    implementation(project(":core-contracts"))
    implementation(project(":core-pipeline"))
    implementation(project(":core-motion"))
    implementation(project(":core-alignment"))
    testImplementation(kotlin("test"))
    testImplementation(libs.kotlinx.coroutines.core)
}

tasks.test {
    useJUnitPlatform()
}
