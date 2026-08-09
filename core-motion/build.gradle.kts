plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(17)
    compilerOptions {
        allWarningsAsErrors.set(true)
    }
}

dependencies {
    implementation(project(":core-contracts"))
    implementation(project(":core-pipeline"))
    testImplementation(kotlin("test"))
    testImplementation(libs.kotlinx.coroutines.core)
    testImplementation(libs.kotlinx.serialization.json)
}

sourceSets.test {
    resources.srcDir(project(":core-contracts").file("src/test/resources"))
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("failed", "skipped")
        showExceptions = true
        showCauses = true
        showStackTraces = true
    }
}

val committedMotionResources = layout.projectDirectory.dir("src/test/resources")

val updateMotionArtifacts by tasks.registering(JavaExec::class) {
    group = "verification"
    description = "Regenerate committed traces and the native MP33 parity fixture."
    dependsOn(tasks.testClasses)
    classpath = sourceSets.test.get().runtimeClasspath
    mainClass.set("ai.senp.motion.MotionArtifactTool")
    args("update", committedMotionResources.asFile.absolutePath)
    outputs.upToDateWhen { false }
}

val checkMotionArtifacts by tasks.registering(JavaExec::class) {
    group = "verification"
    description = "Fail when committed motion traces or fixtures differ from deterministic generation."
    dependsOn(tasks.testClasses)
    classpath = sourceSets.test.get().runtimeClasspath
    mainClass.set("ai.senp.motion.MotionArtifactTool")
    val generated = layout.buildDirectory.dir("generated/motion-artifacts")
    args("check", committedMotionResources.asFile.absolutePath, generated.get().asFile.absolutePath)
    outputs.dir(generated)
    outputs.upToDateWhen { false }
}

val microbenchmark by tasks.registering(JavaExec::class) {
    group = "verification"
    description = "Run the deterministic synthetic ten-second motion-core microbenchmark."
    dependsOn(tasks.testClasses)
    classpath = sourceSets.test.get().runtimeClasspath
    mainClass.set("ai.senp.motion.Microbenchmark")
    val report = layout.buildDirectory.file("reports/microbenchmark/motion-core-10s.json")
    args(report.get().asFile.absolutePath, "120")
    outputs.file(report)
    outputs.upToDateWhen { false }
}

val spatialMicrobenchmark by tasks.registering(JavaExec::class) {
    group = "verification"
    description = "Run the synchronization-v2 spatial kernel on a deterministic ten-second 15 FPS pose pair."
    dependsOn(tasks.testClasses)
    classpath = sourceSets.test.get().runtimeClasspath
    mainClass.set("ai.senp.motion.SpatialMicrobenchmark")
    val report = layout.buildDirectory.file("reports/microbenchmark/spatial-sync-v2-10s.json")
    args(report.get().asFile.absolutePath, "120")
    outputs.file(report)
    outputs.upToDateWhen { false }
}

tasks.register("verifySpatialSynchronization") {
    group = "verification"
    description = "Run motion tests plus the synchronization-v2 spatial microbenchmark."
    dependsOn(tasks.test, spatialMicrobenchmark)
}

tasks.register("verifyMotionCore") {
    group = "verification"
    description = "Run tests, verify committed artifacts, and execute the ten-second synthetic benchmark."
    dependsOn(tasks.test, checkMotionArtifacts, microbenchmark)
}
