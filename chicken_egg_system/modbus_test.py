import asyncio
from pymodbus.server import StartAsyncTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext

async def run_server():
    store = ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [0] * 100),
        co=ModbusSequentialDataBlock(0, [0] * 100),
        hr=ModbusSequentialDataBlock(0, [0] * 100),
        ir=ModbusSequentialDataBlock(0, [0] * 100)
    )
    context = ModbusServerContext(slaves=store, single=True)

    identity = ModbusDeviceIdentification()
    identity.VendorName = "My Modbus Server"
    identity.ProductCode = "PM"
    identity.VendorUrl = "http://example.com"
    identity.ProductName = "Modbus Server"
    identity.ModelName = "PM-MB"
    identity.MajorMinorRevision = "1.0"

    print("Modbus TCP Slave đang chạy trên cổng 5020...")
    await StartAsyncTcpServer(
        context=context,
        identity=identity,
        address=("0.0.0.0", 5020)
    )

if __name__ == "__main__":
    asyncio.run(run_server())