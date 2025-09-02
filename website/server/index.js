import express from 'express';
import cors from 'cors';
import { AthenaClient, ListDataCatalogsCommand, ListDatabasesCommand, ListTableMetadataCommand } from '@aws-sdk/client-athena';
import { GlueClient, GetPartitionsCommand, GetTableCommand } from '@aws-sdk/client-glue';

const app = express();
app.use(express.json());
app.use(cors());

const PORT = process.env.PORT || 8787;

const region = process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION || 'us-east-1';
const athena = new AthenaClient({ region });
const glue = new GlueClient({ region });

app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

app.get('/api/data-sources', async (_req, res) => {
  try {
    const out = await athena.send(new ListDataCatalogsCommand({}));
    const names = (out.DataCatalogsSummary || []).map(d => d?.CatalogName).filter(Boolean);
    if (!names.includes('AwsDataCatalog')) names.unshift('AwsDataCatalog');
    res.json({ dataSources: names });
  } catch (e) {
    res.status(500).json({ error: e?.message || 'Failed to list data sources' });
  }
});

app.get('/api/catalogs', async (req, res) => {
  try {
    const { dataSource } = req.query;
    if (!dataSource) return res.status(400).json({ error: 'Missing dataSource' });
    // For Athena, the data source corresponds to a single catalog of the same name
    res.json({ catalogs: [String(dataSource)] });
  } catch (e) {
    res.status(500).json({ error: e?.message || 'Failed to list catalogs' });
  }
});

app.get('/api/databases', async (req, res) => {
  try {
    const { catalog } = req.query;
    if (!catalog) return res.status(400).json({ error: 'Missing catalog' });
    const out = await athena.send(new ListDatabasesCommand({ CatalogName: String(catalog) }));
    const dbs = (out.DatabaseList || []).map(d => d?.Name).filter(Boolean);
    res.json({ databases: dbs });
  } catch (e) {
    res.status(500).json({ error: e?.message || 'Failed to list databases' });
  }
});

app.get('/api/tables', async (req, res) => {
  try {
    const { catalog, database } = req.query;
    if (!catalog || !database) return res.status(400).json({ error: 'Missing catalog or database' });
    const out = await athena.send(new ListTableMetadataCommand({ CatalogName: String(catalog), DatabaseName: String(database) }));
    const tables = (out.TableMetadataList || []).map(t => ({ name: t?.Name })).filter(t => t.name);
    res.json({ tables });
  } catch (e) {
    res.status(500).json({ error: e?.message || 'Failed to list tables' });
  }
});

app.get('/api/partitions', async (req, res) => {
  try {
    const { catalog, database, table } = req.query;
    if (!catalog || !database || !table) return res.status(400).json({ error: 'Missing catalog, database or table' });

    // Get partition keys
    const tableResp = await glue.send(new GetTableCommand({
      CatalogId: undefined, // default account
      DatabaseName: String(database),
      Name: String(table),
    }));
    const partitionKeys = (tableResp.Table?.PartitionKeys || []).map(k => k?.Name).filter(Boolean);

    if (!partitionKeys.length) {
      return res.json({ partitionKeys: [], partitions: [] });
    }

    // List partitions (first page)
    const partResp = await glue.send(new GetPartitionsCommand({
      DatabaseName: String(database),
      TableName: String(table),
      MaxResults: 500,
    }));

    const partitions = (partResp.Partitions || []).map(p => {
      const obj = {};
      (p.Values || []).forEach((val, idx) => {
        const key = partitionKeys[idx];
        if (key) obj[key] = val;
      });
      return obj;
    });

    res.json({ partitionKeys, partitions });
  } catch (e) {
    res.status(500).json({ error: e?.message || 'Failed to list partitions' });
  }
});

app.listen(PORT, () => {
  console.log(`[server] listening on http://localhost:${PORT} (region: ${region}, profile: ${process.env.AWS_PROFILE || 'default'})`);
});


