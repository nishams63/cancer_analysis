import { PrismaClient } from "@prisma/client";
const prisma = new PrismaClient();

async function main() {
  await prisma.patient.upsert({ where: { id: "ONC-2048" }, update: {}, create: { id: "ONC-2048", age: 62, sex: "Male", cancerType: "NSCLC", stage: "IV", ecog: 1, metadata: { synthetic: true, treatment: "Carboplatin + Pemetrexed" } } });
  await prisma.user.upsert({ where: { email: "dr.sharma@onco.ai" }, update: {}, create: { email: "dr.sharma@onco.ai", name: "Dr. Priya Sharma", role: "oncologist" } });
}
main().finally(() => prisma.$disconnect());
